#![cfg(feature = "gpu-check")]

use ash::{vk, Entry};
use shadowbox::interface::ShaderInput;
use shadowbox::ir::ops;
use std::{ffi::CString, fs, path::Path, process::Command};

fn compile(source: &str, stage: &str, path: &Path) -> Vec<u32> {
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

// Maxwell encodings: envytools gm107.c:938,951,1965,1967;
// Mesa sm50.rs:2223-2236. Sample register semantics: Ryujinx
// InstEmitTexture.cs:294-299,536-542; SPIR-V shape: Instructions.cs:1394-1415.
fn snippet(scalar: bool, sample: u32) -> Vec<u32> {
    let pt = 7u64 << 16;
    // Mesa sm50.rs:1938-1941: setup literals enable all quad positions.
    let mov = |r: u64, v: u32| 0x0100u64 << 48 | u64::from(v) << 20 | 15 << 12 | pt | r;
    let mut slots = vec![
        mov(8, 1),
        mov(9, 2),
        mov(20, if scalar { sample } else { 1 }),
        mov(21, sample),
    ];
    slots.push(if scalar {
        0xda00u64 << 48 | 6 << 53 | 4 << 50 | 7 << 36 | 2 << 28 | 20 << 20 | pt | 8 << 8
    } else {
        0xdd38u64 << 48 | 1 << 50 | 15 << 31 | 2 << 28 | 20 << 20 | pt | 8 << 8
    });
    slots.push(0xe300u64 << 48 | pt | 0xf);
    let code = slots
        .chunks(3)
        .flat_map(|s| [0, s[0], s[1], s[2]])
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>();
    let mut header = vec![0; 128];
    header[1] = 5 << 2;
    header[72] = 15;
    Result::from(shadowbox::translate(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &code,
            program_record: &header,
        },
    ))
    .unwrap()
    .spirv
}

// TXQ bound encoding: Mesa sm50.rs:2351-2381; envytools gm107.c:1958,882-891.
// Sample count Z: Ryujinx InstEmitTexture.cs:1152-1182 and
// CodeGen/Spirv/Instructions.cs:1460-1473.
fn bound_sample_count_query(optimized: bool) -> Vec<u32> {
    use shadowbox::{
        decoder::decode_word,
        emitter::emit_unoptimized_with_graphics_state,
        headers,
        interface::{BoundTexture, GraphicsState, TextureDimension},
        ir::{
            Conversion, Instruction, InstructionKind as K, Register as R, RoundingMode, Source as S,
        },
    };
    let raw = 0xdf48u64 << 48 | 7 << 36 | 4 << 31 | 2 << 22 | 7 << 16 | 255 << 8 | 8;
    let mut xs = vec![decode_word([raw as u32, (raw >> 32) as u32], 0)];
    for dst in 0..4 {
        xs.push(Instruction {
            address: xs.len() as u32 * 8,
            predicate: None,
            kind: K::Convert(ops::Convert {
                kind: Conversion::I2F {
                    signed: false,
                    dst_size: 2,
                    src_size: 2,
                    round: RoundingMode::NearestEven,
                    byte: 0,
                    abs: false,
                    negate: false,
                },
                dst: R(dst),
                src: S::Register(R(8)),
            }),
        });
    }
    xs.push(Instruction {
        address: xs.len() as u32 * 8,
        predicate: None,
        kind: K::Exit(ops::Exit { cc_test: 15 }),
    });
    let mut words = [0; 32];
    words[0] = 5 << 10;
    words[18] = 15;
    let header = headers::parse(&words).unwrap();
    let bindings = [BoundTexture {
        binding: 7,
        dimension: TextureDimension::D2,
        depth_compare: false,
        multisample: true,
    }];
    let state = GraphicsState {
        texture_bindings: &bindings,
        ..Default::default()
    };
    let mut module = emit_unoptimized_with_graphics_state(&xs, Some(&header), &state).unwrap();
    if optimized {
        module.optimize();
    }
    module.words
}

#[test]
fn multisample_fetch_selects_each_sample_exactly_on_radv() {
    let temp = std::env::temp_dir().join(format!("shadowbox-gpu-ms-check-{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    let vert = compile(
        "#version 450\nvoid main(){vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex],0,1);}",
        "vert", &temp.join("fullscreen"));
    let fill = compile("#version 450\nlayout(location=0) out vec4 color; void main(){float s=float(gl_SampleID); color=vec4(s+0.125*floor(gl_FragCoord.x),0.25*floor(gl_FragCoord.y),s*0.25,-s);}", "frag", &temp.join("fill"));
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
        let physical = physical.expect("RADV GPU required for an exact multisample check");
        let props = instance.get_physical_device_queue_family_properties(physical);
        let family = props
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap() as u32;
        let priority = [1.0];
        let device = instance
            .create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .enabled_features(
                        &vk::PhysicalDeviceFeatures::default().sample_rate_shading(true),
                    )
                    .queue_create_infos(std::slice::from_ref(
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
        let tex_image = device
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
                    .samples(vk::SampleCountFlags::TYPE_4)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .unwrap();
        let tex_req = device.get_image_memory_requirements(tex_image);
        let tex_mem = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(tex_req.size)
                    .memory_type_index(find_mem(
                        tex_req.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )),
                None,
            )
            .unwrap();
        device.bind_image_memory(tex_image, tex_mem, 0).unwrap();
        let tex_view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(tex_image)
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
        let tex_render = device
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&[vk::AttachmentDescription::default()
                        .format(vk::Format::R32G32B32A32_SFLOAT)
                        .samples(vk::SampleCountFlags::TYPE_4)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)])
                    .dependencies(&[vk::SubpassDependency::default()
                        .src_subpass(0)
                        .dst_subpass(vk::SUBPASS_EXTERNAL)
                        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)])
                    .subpasses(&[vk::SubpassDescription::default()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&[vk::AttachmentReference {
                            attachment: 0,
                            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        }])]),
                None,
            )
            .unwrap();
        let tex_framebuffer = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(tex_render)
                    .attachments(&[tex_view])
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
        let fs_fill = make_shader(&fill);
        let descriptor_layouts = [0, 14].map(|binding| {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(binding)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    ]),
                    None,
                )
                .unwrap()
        });
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(2)
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: 2,
                    }]),
                None,
            )
            .unwrap();
        let descriptors = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&descriptor_layouts),
            )
            .unwrap();
        for (set, binding) in descriptors.iter().zip([0, 14]) {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .image_view(tex_view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)])],
                &[],
            );
        }
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_layouts),
                None,
            )
            .unwrap();
        let run = |fragment, populate: bool| {
            let render = if populate { tex_render } else { render };
            let framebuffer = if populate {
                tex_framebuffer
            } else {
                framebuffer
            };
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
                                .rasterization_samples(if populate {
                                    vk::SampleCountFlags::TYPE_4
                                } else {
                                    vk::SampleCountFlags::TYPE_1
                                })
                                .sample_shading_enable(populate)
                                .min_sample_shading(1.0),
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
            if !populate {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &descriptors,
                    &[],
                );
            }
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
            if !populate {
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
            }
            device.end_command_buffer(cmd).unwrap();
            device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                    vk::Fence::null(),
                )
                .unwrap();
            device.queue_wait_idle(queue).unwrap();
            let bytes = if populate {
                Vec::new()
            } else {
                let bytes = std::slice::from_raw_parts(
                    device
                        .map_memory(bmem, 0, 256, vk::MemoryMapFlags::empty())
                        .unwrap() as *const u8,
                    256,
                )
                .to_vec();
                device.unmap_memory(bmem);
                bytes
            };
            device.destroy_pipeline(pipeline, None);
            device.destroy_command_pool(pool, None);
            bytes
        };
        run(fs_fill, true);
        for scalar in [false, true] {
            for sample in 0..4 {
                let reference = compile(&format!("#version 450\n#extension GL_EXT_samplerless_texture_functions : require\nlayout(set=0,binding=0) uniform texture2DMS tex; layout(location=0) out vec4 color; void main(){{color=texelFetch(tex,ivec2(1,2),{sample});}}"), "frag", &temp.join("reference"));
                let words = snippet(scalar, sample);
                let path = temp.join("translated.spv");
                fs::write(
                    &path,
                    words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                assert!(Command::new("spirv-val")
                    .args(["--target-env", "vulkan1.2"])
                    .arg(path)
                    .status()
                    .unwrap()
                    .success());
                let fs_ref = make_shader(&reference);
                let fs_shadow = make_shader(&words);
                let a = run(fs_ref, false);
                let b = run(fs_shadow, false);
                assert_eq!(a, b, "scalar {scalar}, sample {sample}");
                let s = sample as f32;
                let expected = [s + 0.125, 0.5, s * 0.25, -s];
                let expected_bits = expected.map(f32::to_bits);
                let first_pixel = b[..16]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect::<Vec<_>>();
                assert_eq!(first_pixel, expected_bits, "sample contents must differ");
                eprintln!(
                    "{} sample {sample}: exact MATCH",
                    if scalar { "TLDS.MS" } else { "TLD.MS" }
                );
                for shader in [fs_shadow, fs_ref] {
                    device.destroy_shader_module(shader, None);
                }
            }
        }
        // Independent GLSL reports the actual sample count of the same image.
        let reference = compile("#version 450\n#extension GL_EXT_samplerless_texture_functions : require\nlayout(set=1,binding=14) uniform texture2DMS tex; layout(location=0) out vec4 color; void main(){color=vec4(float(textureSamples(tex)));}", "frag", &temp.join("query-reference"));
        let fs_ref = make_shader(&reference);
        let expected = run(fs_ref, false);
        assert!(expected
            .as_chunks::<4>()
            .0
            .iter()
            .all(|w| u32::from_le_bytes(*w) == 4.0f32.to_bits()));
        for optimized in [false, true] {
            let words = bound_sample_count_query(optimized);
            let path = temp.join("query.spv");
            fs::write(
                &path,
                words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            assert!(Command::new("spirv-val")
                .args(["--target-env", "vulkan1.2"])
                .arg(&path)
                .status()
                .unwrap()
                .success());
            let fs_shadow = make_shader(&words);
            assert_eq!(
                run(fs_shadow, false),
                expected,
                "TXQ.samples optimized={optimized}"
            );
            eprintln!("TXQ.samples optimized={optimized}: 64 exact words, 0 differences");
            device.destroy_shader_module(fs_shadow, None);
        }
        device.destroy_shader_module(fs_ref, None);
        for shader in [fs_fill, vs] {
            device.destroy_shader_module(shader, None);
        }
        device.destroy_descriptor_pool(descriptor_pool, None);
        for descriptor in descriptor_layouts {
            device.destroy_descriptor_set_layout(descriptor, None);
        }
        device.destroy_framebuffer(tex_framebuffer, None);
        device.destroy_render_pass(tex_render, None);
        device.destroy_image_view(tex_view, None);
        device.destroy_image(tex_image, None);
        device.free_memory(tex_mem, None);
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
    fs::remove_dir_all(temp).unwrap();
}
