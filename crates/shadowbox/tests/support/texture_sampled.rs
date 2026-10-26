// Project-owned immutable test image, with distinct channels and mip levels.
// Vulkan resource transport only. No instruction semantics are defined here.
use ash::vk;
pub struct Texture {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
}
impl Texture {
    pub unsafe fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        command: vk::CommandBuffer,
        queue: vk::Queue,
        shape: (u32, bool, bool),
    ) -> Self {
        unsafe {
            let (dim, arrayed, depth) = shape;
            let layers = if dim == 3 {
                if arrayed {
                    12
                } else {
                    6
                }
            } else if arrayed {
                3
            } else {
                1
            };
            let extent = vk::Extent3D {
                width: 4,
                height: if dim == 0 { 1 } else { 4 },
                depth: if dim == 2 { 4 } else { 1 },
            };
            let format = if depth {
                vk::Format::D32_SFLOAT
            } else {
                vk::Format::R32G32B32A32_SFLOAT
            };
            let aspect = if depth {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };
            let channels = if depth { 1 } else { 4 };
            let mut data = Vec::new();
            let mut regions = Vec::new();
            for level in 0..2 {
                let size = vk::Extent3D {
                    width: extent.width >> level,
                    height: (extent.height >> level).max(1),
                    depth: (extent.depth >> level).max(1),
                };
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset((data.len() * 4) as u64)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(aspect)
                                .mip_level(level)
                                .layer_count(layers),
                        )
                        .image_extent(size),
                );
                for layer in 0..layers {
                    for z in 0..size.depth {
                        for y in 0..size.height {
                            for x in 0..size.width {
                                for channel in 0..channels {
                                    let numerator = 1
                                        + layer * 512
                                        + level * 128
                                        + z * 128
                                        + y * 32
                                        + x * 4
                                        + channel;
                                    let divisor = if depth { 8192.0 } else { 256.0 };
                                    data.push((numerator as f32 / divisor).to_bits());
                                }
                            }
                        }
                    }
                }
            }
            let bytes = (data.len() * 4) as u64;
            let properties = instance.get_physical_device_memory_properties(physical);
            let image = device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .flags(if dim == 3 {
                            vk::ImageCreateFlags::CUBE_COMPATIBLE
                        } else {
                            vk::ImageCreateFlags::empty()
                        })
                        .image_type(match dim {
                            0 => vk::ImageType::TYPE_1D,
                            2 => vk::ImageType::TYPE_3D,
                            _ => vk::ImageType::TYPE_2D,
                        })
                        .format(format)
                        .extent(extent)
                        .mip_levels(2)
                        .array_layers(layers)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
                    None,
                )
                .unwrap();
            let req = device.get_image_memory_requirements(image);
            let ty = (0..properties.memory_type_count)
                .find(|i| req.memory_type_bits & (1 << i) != 0)
                .unwrap();
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(ty),
                    None,
                )
                .unwrap();
            device.bind_image_memory(image, memory, 0).unwrap();
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(bytes)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    None,
                )
                .unwrap();
            let req = device.get_buffer_memory_requirements(buffer);
            let ty = (0..properties.memory_type_count)
                .find(|i| {
                    req.memory_type_bits & (1 << i) != 0
                        && properties.memory_types[*i as usize]
                            .property_flags
                            .contains(
                                vk::MemoryPropertyFlags::HOST_VISIBLE
                                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                            )
                })
                .unwrap();
            let upload = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(ty),
                    None,
                )
                .unwrap();
            device.bind_buffer_memory(buffer, upload, 0).unwrap();
            let mapped = device
                .map_memory(upload, 0, bytes, vk::MemoryMapFlags::empty())
                .unwrap()
                .cast::<u32>();
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped, data.len());
            device.unmap_memory(upload);
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .level_count(2)
                .layer_count(layers);
            device
                .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
                .unwrap();
            device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(range)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
            );
            device.cmd_copy_buffer_to_image(
                command,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::ALL_GRAPHICS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(range)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            );
            device.end_command_buffer(command).unwrap();
            super::submit_and_wait(device, queue, command);
            device.destroy_buffer(buffer, None);
            device.free_memory(upload, None);
            let view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(match (dim, arrayed) {
                            (0, false) => vk::ImageViewType::TYPE_1D,
                            (0, true) => vk::ImageViewType::TYPE_1D_ARRAY,
                            (1, false) => vk::ImageViewType::TYPE_2D,
                            (1, true) => vk::ImageViewType::TYPE_2D_ARRAY,
                            (2, false) => vk::ImageViewType::TYPE_3D,
                            (3, false) => vk::ImageViewType::CUBE,
                            (3, true) => vk::ImageViewType::CUBE_ARRAY,
                            _ => panic!("unsupported sampled image shape"),
                        })
                        .format(format)
                        .subresource_range(range),
                    None,
                )
                .unwrap();
            let sampler = device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .max_lod(1.0)
                        .compare_enable(depth)
                        .compare_op(vk::CompareOp::LESS_OR_EQUAL)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .unwrap();
            Self {
                image,
                memory,
                view,
                sampler,
            }
        }
    }
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
