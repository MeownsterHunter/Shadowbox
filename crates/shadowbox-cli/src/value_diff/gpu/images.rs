// Project-owned upload and sampler inputs. Image, view, sampler, copy and
// barrier fields: /usr/include/vulkan/vulkan_core.h:3898-3948,3996-4007,
// 4025-4036,4140-4159. Type/view/format enums:1968,1985,2225-2229,2327-2333;
// nearest, clamp and comparison enums:2373,2381-2406; cube flag:2766; layouts:2275-2277; aspects:2966-2984; descriptor info:4179-4183.
// Sampling types follow Ryujinx CodeGen/Spirv/Declarations.cs:168-194.
// This module does not translate shader instructions or copy upstream code.
use super::{error, Gpu};
use crate::value_diff::module::{images::ResourceKind, Plan};
use ash::vk;

pub(super) struct ImageAllocation {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    sampler: vk::Sampler,
    staging: Option<usize>,
}

pub(super) unsafe fn destroy(gpu: &Gpu) {
    for a in &gpu.images {
        gpu.device.destroy_sampler(a.sampler, None);
        gpu.device.destroy_image_view(a.view, None);
        gpu.device.destroy_image(a.image, None);
        gpu.device.free_memory(a.memory, None);
    }
}

pub(super) unsafe fn create(
    gpu: &mut Gpu,
    plan: &Plan,
    memory: &vk::PhysicalDeviceMemoryProperties,
    physical: vk::PhysicalDevice,
) -> Result<(), String> {
    for r in &plan.images {
        gpu.images.push(ImageAllocation {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            sampler: vk::Sampler::null(),
            staging: None,
        });
        let index = gpu.images.len() - 1;
        let (needs_sampler, compare) = match r.kind {
            ResourceKind::Image { ty, combined } => (combined, ty.depth),
            ResourceKind::Sampler { compare } => (true, compare.unwrap_or(false)),
        };
        if needs_sampler {
            gpu.images[index].sampler = gpu
                .device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .compare_enable(compare)
                        .compare_op(vk::CompareOp::LESS_OR_EQUAL)
                        .min_lod(0.0)
                        .max_lod(0.0),
                    None,
                )
                .map_err(error)?;
        }
        let ResourceKind::Image { ty, .. } = r.kind else {
            continue;
        };
        let image_type = match ty.dim {
            0 => vk::ImageType::TYPE_1D,
            2 => vk::ImageType::TYPE_3D,
            _ => vk::ImageType::TYPE_2D,
        };
        let flags = if ty.dim == 3 {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        };
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        // Ask about the exact image tuple before allocating, so unsupported
        // depth/dimension combinations fail without submitting GPU work.
        // Format-properties query: vulkan_core.h:3497-3503,4535.
        let supported = gpu
            .instance
            .get_physical_device_image_format_properties(
                physical,
                ty.format(),
                image_type,
                vk::ImageTiling::OPTIMAL,
                usage,
                flags,
            )
            .map_err(|e| {
                format!(
                    "sampled image format {:?} {:?} unavailable: {e:?}",
                    ty.format(),
                    ty.view()
                )
            })?;
        let extent = ty.extent();
        if extent.width > supported.max_extent.width
            || extent.height > supported.max_extent.height
            || extent.depth > supported.max_extent.depth
            || ty.layers() > supported.max_array_layers
            || !supported
                .sample_counts
                .contains(vk::SampleCountFlags::TYPE_1)
        {
            return Err("sampled image extent unavailable".into());
        }
        let handle = gpu
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .flags(flags)
                    .image_type(image_type)
                    .format(ty.format())
                    .extent(ty.extent())
                    .mip_levels(1)
                    .array_layers(ty.layers())
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(error)?;
        gpu.images[index].image = handle;
        let req = gpu.device.get_image_memory_requirements(handle);
        let memory_type = (0..memory.memory_type_count)
            .find(|&i| req.memory_type_bits & (1 << i) != 0)
            .ok_or("no image memory type")?;
        let allocation = gpu
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memory_type),
                None,
            )
            .map_err(error)?;
        gpu.images[index].memory = allocation;
        gpu.device
            .bind_image_memory(handle, allocation, 0)
            .map_err(error)?;
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(ty.aspect())
            .level_count(1)
            .layer_count(ty.layers());
        gpu.images[index].view = gpu
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(handle)
                    .view_type(ty.view())
                    .format(ty.format())
                    .subresource_range(range),
                None,
            )
            .map_err(error)?;
        let staging = gpu.allocate(
            memory,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &ty.texels(r.set, r.binding),
        )?;
        gpu.images[index].staging = Some(staging);
    }
    Ok(())
}

pub(super) unsafe fn descriptors(gpu: &Gpu, plan: &Plan, sets: &[vk::DescriptorSet]) {
    for (r, a) in plan.images.iter().zip(&gpu.images) {
        let info = [vk::DescriptorImageInfo::default()
            .sampler(a.sampler)
            .image_view(a.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        gpu.device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(sets[r.set as usize])
                .dst_binding(r.binding)
                .descriptor_type(r.kind.descriptor())
                .image_info(&info)],
            &[],
        );
    }
}

pub(super) unsafe fn upload(gpu: &Gpu, plan: &Plan, command: vk::CommandBuffer) {
    for (r, a) in plan.images.iter().zip(&gpu.images) {
        let ResourceKind::Image { ty, .. } = r.kind else {
            continue;
        };
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(ty.aspect())
            .level_count(1)
            .layer_count(ty.layers());
        let barrier = vk::ImageMemoryBarrier::default()
            .image(a.image)
            .subresource_range(range)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        gpu.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
        let copy = vk::BufferImageCopy::default()
            .image_extent(ty.extent())
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(ty.aspect())
                    .layer_count(ty.layers()),
            );
        gpu.device.cmd_copy_buffer_to_image(
            command,
            gpu.allocations[a.staging.unwrap()].buffer,
            a.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy],
        );
        let barrier = barrier
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        gpu.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}
