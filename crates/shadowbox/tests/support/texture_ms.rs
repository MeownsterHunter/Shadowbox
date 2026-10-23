// Project-owned per-sample patterns, initialized by a separate GLSL compute
// shader. No Maxwell semantics are used by this resource transport.
use super::texture::Texture;
use ash::vk;
use std::{fs, process::Command};

pub unsafe fn create(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    command: vk::CommandBuffer,
    queue: vk::Queue,
    arrayed: bool,
) -> Texture {
    unsafe {
        let layers = if arrayed { 3 } else { 1 };
        let format = vk::Format::R32G32B32A32_SFLOAT;
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: 4,
                        height: 4,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(layers)
                    .samples(vk::SampleCountFlags::TYPE_4)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
                None,
            )
            .unwrap();
        let properties = instance.get_physical_device_memory_properties(physical);
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
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(layers);
        let view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .format(format)
                    .view_type(if arrayed {
                        vk::ImageViewType::TYPE_2D_ARRAY
                    } else {
                        vk::ImageViewType::TYPE_2D
                    })
                    .subresource_range(range),
                None,
            )
            .unwrap();
        let sampler = device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .min_filter(vk::Filter::NEAREST)
                    .mag_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST),
                None,
            )
            .unwrap();
        let dir = std::path::PathBuf::from(
            std::env::var_os("CARGO_TARGET_DIR").expect("external build directory"),
        )
        .join(format!("ms-patterns-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join(format!(
            "init-{}-{:?}-{arrayed}.comp",
            std::process::id(),
            std::thread::current().id()
        ));
        let path = source.with_extension("spv");
        fs::write(
            &source,
            format!(
                "#version 450
layout(local_size_x=4,local_size_y=4) in;
layout(set=0,binding=0,rgba32f) uniform image2DMS{suffix} img;
void main(){{ivec3 p=ivec3(gl_GlobalInvocationID);
for(int s=0;s<4;s++){{float v=float(1+s*128+p.z*2048+p.y*32+p.x*4);
imageStore(img,{position},s,(vec4(v)+vec4(0,1,2,3))/256.0);}}
}}",
                suffix = if arrayed { "Array" } else { "" },
                position = if arrayed { "p" } else { "p.xy" }
            ),
        )
        .unwrap();
        let compile = Command::new("glslc")
            .args(["--target-env=vulkan1.2", "-fshader-stage=compute"])
            .arg(&source)
            .arg("-o")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "{}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let code = fs::read(path)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect::<Vec<_>>();
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let descriptor = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )
            .unwrap();
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[descriptor]),
                None,
            )
            .unwrap();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                    }]),
                None,
            )
            .unwrap();
        let set = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&[descriptor]),
            )
            .unwrap()[0];
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&[vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)])],
            &[],
        );
        let shader = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .unwrap();
        let pipeline = device
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
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)],
        );
        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[set],
            &[],
        );
        device.cmd_dispatch(command, 1, 1, layers);
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .image(image)
                .subresource_range(range)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
        );
        device.end_command_buffer(command).unwrap();
        super::submit_and_wait(device, queue, command);
        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(shader, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(descriptor, None);
        Texture {
            image,
            memory,
            view,
            sampler,
        }
    }
}
