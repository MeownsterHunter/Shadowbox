// Shared exact fragment transport from gpu_check.rs. Render to RGBA32F and
// compare every component bit at all 16 pixels, including optimized modules.
use ash::{vk, Entry};
use std::{ffi::CString, fs, path::Path, process::Command};

pub fn compile(source: &str, stage: &str, path: &Path) -> Vec<u32> {
    let src = path.with_extension(stage);
    let out = path.with_extension(format!("{stage}.spv"));
    fs::write(&src, source).unwrap();
    assert!(Command::new("glslc")
        .args(["--target-env=vulkan1.2", &format!("-fshader-stage={stage}")])
        .arg(&src)
        .arg("-o")
        .arg(&out)
        .status()
        .unwrap()
        .success());
    fs::read(out)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| u32::from_le_bytes(*x))
        .collect()
}

pub fn assert_pixels(variants: &[(&str, Vec<u32>)], expected: [f32; 4]) {
    let temp = std::env::temp_dir().join(format!(
        "shadowbox-fragment-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("check")
    ));
    fs::create_dir_all(&temp).unwrap();
    let vert = compile(
        "#version 450\nvoid main(){vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex],0,1);}",
        "vert", &temp.join("fullscreen"));
    for (index, (name, words)) in variants.iter().enumerate() {
        let path = temp.join(format!("variant-{index}.spv"));
        fs::write(
            &path,
            words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let result = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    unsafe {
        let entry = Entry::load().unwrap();
        let app = CString::new("shadowbox-gpu-check").unwrap();
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(
                    &vk::ApplicationInfo::default()
                        .application_name(&app)
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
            });
        let physical = physical.expect("RADV GPU required for exact fragment readback");
        let props = instance.get_physical_device_queue_family_properties(physical);
        let family = props
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap() as u32;
        let priority = [1.0];
        let device = instance
            .create_device(
                physical,
                &vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(
                    &vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(family)
                        .queue_priorities(&priority),
                )),
                None,
            )
            .unwrap();
        let queue = device.get_device_queue(family, 0);
        let memory = instance.get_physical_device_memory_properties(physical);
        let find_mem = |bits, flags| {
            (0..memory.memory_type_count)
                .find(|i| {
                    bits & (1 << i) != 0
                        && memory.memory_types[*i as usize]
                            .property_flags
                            .contains(flags)
                })
                .unwrap()
        };
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
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .unwrap();
        let req = device.get_image_memory_requirements(image);
        let imem = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(find_mem(
                        req.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )),
                None,
            )
            .unwrap();
        device.bind_image_memory(image, imem, 0).unwrap();
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(256)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
            .unwrap();
        let breq = device.get_buffer_memory_requirements(buffer);
        let bmem = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(breq.size)
                    .memory_type_index(find_mem(
                        breq.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )),
                None,
            )
            .unwrap();
        device.bind_buffer_memory(buffer, bmem, 0).unwrap();
        let view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
            .unwrap();
        let render = device
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&[vk::AttachmentDescription::default()
                        .format(vk::Format::R32G32B32A32_SFLOAT)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)])
                    .subpasses(&[vk::SubpassDescription::default()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&[vk::AttachmentReference {
                            attachment: 0,
                            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        }])]),
                None,
            )
            .unwrap();
        let framebuffer = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render)
                    .attachments(&[view])
                    .width(4)
                    .height(4)
                    .layers(1),
                None,
            )
            .unwrap();
        let make_shader = |words: &[u32]| {
            device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
                .unwrap()
        };
        let vs = make_shader(&vert);
        let shaders = variants
            .iter()
            .map(|(_, words)| make_shader(words))
            .collect::<Vec<_>>();
        let layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
            .unwrap();
        let run = |fragment| {
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vs)
                    .name(c"main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment)
                    .name(c"main"),
            ];
            let pipeline = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&stages)
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewports(&[vk::Viewport {
                                    x: 0.0,
                                    y: 0.0,
                                    width: 4.0,
                                    height: 4.0,
                                    min_depth: 0.0,
                                    max_depth: 1.0,
                                }])
                                .scissors(&[vk::Rect2D {
                                    offset: vk::Offset2D::default(),
                                    extent: vk::Extent2D {
                                        width: 4,
                                        height: 4,
                                    },
                                }]),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .polygon_mode(vk::PolygonMode::FILL)
                                .cull_mode(vk::CullModeFlags::NONE)
                                .line_width(1.0),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)]),
                        )
                        .layout(layout)
                        .render_pass(render)],
                    None,
                )
                .unwrap()[0];
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                    None,
                )
                .unwrap();
            let cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0];
            device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(render)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: vk::Extent2D {
                            width: 4,
                            height: 4,
                        },
                    })
                    .clear_values(&[vk::ClearValue {
                        color: vk::ClearColorValue { float32: [0.0; 4] },
                    }]),
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &[vk::BufferImageCopy::default()
                    .buffer_row_length(4)
                    .buffer_image_height(4)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: 4,
                        height: 4,
                        depth: 1,
                    })],
            );
            device.end_command_buffer(cmd).unwrap();
            device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                    vk::Fence::null(),
                )
                .unwrap();
            device.queue_wait_idle(queue).unwrap();
            let bytes = std::slice::from_raw_parts(
                device
                    .map_memory(bmem, 0, 256, vk::MemoryMapFlags::empty())
                    .unwrap() as *const u8,
                256,
            )
            .to_vec();
            device.unmap_memory(bmem);
            device.destroy_pipeline(pipeline, None);
            device.destroy_command_pool(pool, None);
            bytes
        };
        let expected = expected.map(f32::to_bits);
        for ((name, _), &shader) in variants.iter().zip(&shaders) {
            let bytes = run(shader);
            let words = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect::<Vec<_>>();
            for (pixel, rgba) in words.as_chunks::<4>().0.iter().enumerate() {
                assert_eq!(*rgba, expected, "{name}: pixel {pixel}");
            }
            device.destroy_shader_module(shader, None);
        }
        device.destroy_shader_module(vs, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render, None);
        device.destroy_image_view(view, None);
        device.destroy_image(image, None);
        device.free_memory(imem, None);
        device.destroy_buffer(buffer, None);
        device.free_memory(bmem, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    fs::remove_dir_all(&temp).unwrap();
}
