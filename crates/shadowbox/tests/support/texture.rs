// Project-owned sampled and writable image transport.
#[path = "texture_sampled.rs"]
mod sampled;
use super::submit_and_wait;
use ash::vk;
pub use sampled::Texture;
// A separate writable image keeps SUST checks from changing sampled fixtures.
// Vulkan-only transport. Initialize every texel with a GLSL dispatch in tests.
pub struct StorageImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    pub view: vk::ImageView,
}
impl StorageImage {
    pub unsafe fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        command: vk::CommandBuffer,
        queue: vk::Queue,
    ) -> Self {
        unsafe {
            let image = device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R32G32B32A32_SFLOAT)
                        .extent(vk::Extent3D {
                            width: 4,
                            height: 4,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::STORAGE),
                    None,
                )
                .unwrap();
            let req = device.get_image_memory_requirements(image);
            let properties = instance.get_physical_device_memory_properties(physical);
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
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R32G32B32A32_SFLOAT)
                        .subresource_range(range),
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
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(range)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
            );
            device.end_command_buffer(command).unwrap();
            device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(&[command])],
                    vk::Fence::null(),
                )
                .unwrap();
            device.queue_wait_idle(queue).unwrap();
            Self {
                image,
                memory,
                view,
            }
        }
    }
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
