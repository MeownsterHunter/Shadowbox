// Vulkan storage transport only. The instruction oracle is surface_reference.rs.
// Fresh resources for each dispatch prevent stores and atomics from contaminating
// later variants. Copy every image byte back, including untouched texels.
use ash::vk;

pub struct Surface {
    image: vk::Image,
    memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub buffer_view: vk::BufferView,
    buffer: vk::Buffer,
    allocation: vk::DeviceMemory,
    mapped: *mut u8,
    bytes: usize,
    extent: vk::Extent3D,
    layers: u32,
}
impl Surface {
    pub unsafe fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        command: vk::CommandBuffer,
        queue: vk::Queue,
        shape: (u32, bool, u32),
    ) -> Self {
        unsafe {
            let (dim, arrayed, format) = shape;
            // SPIR-V image formats: Mesa src/compiler/spirv/spirv.h:302-331.
            // Raw views: Ryujinx InstEmitSurface.cs:709-720,769-780.
            let (format, texel_bytes) = match format {
                0 | 1 => (vk::Format::R32G32B32A32_SFLOAT, 16),
                24 => (vk::Format::R32_SINT, 4),
                28 => (vk::Format::R16_SINT, 2),
                29 => (vk::Format::R8_SINT, 1),
                30 => (vk::Format::R32G32B32A32_UINT, 16),
                33 => (vk::Format::R32_UINT, 4),
                35 => (vk::Format::R32G32_UINT, 8),
                38 => (vk::Format::R16_UINT, 2),
                39 => (vk::Format::R8_UINT, 1),
                _ => panic!("unsupported storage format {format}"),
            };
            let extent = vk::Extent3D {
                width: 64,
                height: if dim == 0 || dim == 5 { 1 } else { 4 },
                depth: if dim == 2 { 4 } else { 1 },
            };
            let layers = if arrayed { 3 } else { 1 };
            let bytes =
                (extent.width * extent.height * extent.depth * layers * texel_bytes) as usize;
            let properties = instance.get_physical_device_memory_properties(physical);
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(bytes as u64).usage(
                        vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                    ),
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
            let allocation = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(ty),
                    None,
                )
                .unwrap();
            device.bind_buffer_memory(buffer, allocation, 0).unwrap();
            let mapped = device
                .map_memory(allocation, 0, req.size, vk::MemoryMapFlags::empty())
                .unwrap()
                .cast::<u8>();
            for i in 0..bytes {
                *mapped.add(i) = (i as u8).wrapping_mul(73).wrapping_add(129);
            }
            if format == vk::Format::R32G32B32A32_SFLOAT {
                for i in 0..bytes / 4 {
                    std::ptr::write_unaligned(
                        mapped.add(i * 4).cast::<u32>(),
                        ((i as f32 + 1.0) / 256.0).to_bits(),
                    );
                }
            }
            let mut result = Self {
                image: vk::Image::null(),
                memory: vk::DeviceMemory::null(),
                view: vk::ImageView::null(),
                buffer_view: vk::BufferView::null(),
                buffer,
                allocation,
                mapped,
                bytes,
                extent,
                layers,
            };
            if dim == 5 {
                result.buffer_view = device
                    .create_buffer_view(
                        &vk::BufferViewCreateInfo::default()
                            .buffer(buffer)
                            .format(format)
                            .range(bytes as u64),
                        None,
                    )
                    .unwrap();
                return result;
            }
            result.image = device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(match dim {
                            0 => vk::ImageType::TYPE_1D,
                            1 => vk::ImageType::TYPE_2D,
                            2 => vk::ImageType::TYPE_3D,
                            _ => panic!("unsupported storage dimension"),
                        })
                        .format(format)
                        .extent(extent)
                        .mip_levels(1)
                        .array_layers(layers)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                        ),
                    None,
                )
                .unwrap();
            let req = device.get_image_memory_requirements(result.image);
            let ty = (0..properties.memory_type_count)
                .find(|i| req.memory_type_bits & (1 << i) != 0)
                .unwrap();
            result.memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(ty),
                    None,
                )
                .unwrap();
            device
                .bind_image_memory(result.image, result.memory, 0)
                .unwrap();
            result.view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(result.image)
                        .format(format)
                        .view_type(match (dim, arrayed) {
                            (0, false) => vk::ImageViewType::TYPE_1D,
                            (0, true) => vk::ImageViewType::TYPE_1D_ARRAY,
                            (1, false) => vk::ImageViewType::TYPE_2D,
                            (1, true) => vk::ImageViewType::TYPE_2D_ARRAY,
                            (2, false) => vk::ImageViewType::TYPE_3D,
                            _ => panic!("unsupported storage view"),
                        })
                        .subresource_range(result.range()),
                    None,
                )
                .unwrap();
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
                    .image(result.image)
                    .subresource_range(result.range())
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
            );
            device.cmd_copy_buffer_to_image(
                command,
                buffer,
                result.image,
                vk::ImageLayout::GENERAL,
                &[result.region()],
            );
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
            device.end_command_buffer(command).unwrap();
            super::submit_and_wait(device, queue, command);
            result
        }
    }
    fn range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(self.layers)
    }
    fn region(&self) -> vk::BufferImageCopy {
        vk::BufferImageCopy::default()
            .image_extent(self.extent)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(self.layers),
            )
    }
    pub unsafe fn copy_back(&self, device: &ash::Device, command: vk::CommandBuffer) {
        unsafe {
            if self.image != vk::Image::null() {
                device.cmd_copy_image_to_buffer(
                    command,
                    self.image,
                    vk::ImageLayout::GENERAL,
                    self.buffer,
                    &[self.region()],
                );
            }
        }
    }
    pub unsafe fn bytes(&self) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(self.mapped, self.bytes).to_vec() }
    }
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            if self.image != vk::Image::null() {
                device.destroy_image_view(self.view, None);
                device.destroy_image(self.image, None);
                device.free_memory(self.memory, None);
            } else {
                device.destroy_buffer_view(self.buffer_view, None);
            }
            device.unmap_memory(self.allocation);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.allocation, None);
        }
    }
}
