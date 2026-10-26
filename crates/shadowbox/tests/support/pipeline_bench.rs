//! Uses the same ash pipeline setup as gpu_check.rs, without draw/readback.
//! Cache disabling follows Mesa radv_instance.c:33, radv_device.c:1138-1144
//! and radv_pipeline_cache.c:189-202. Only vkCreate*Pipelines is timed.
use ash::{vk, Entry};
use shadowbox::spirv::Op;
use std::{collections::BTreeMap, fs, path::Path, time::Instant};

fn words(path: &Path) -> Vec<u32> {
    fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

fn instructions(words: &[u32]) -> impl Iterator<Item = (Op, &[u32])> {
    let mut at = 5;
    std::iter::from_fn(move || {
        if at == words.len() {
            return None;
        }
        let n = (words[at] >> 16) as usize;
        let result = (Op::of(words[at]), &words[at + 1..at + n]);
        at += n;
        Some(result)
    })
}

fn model(words: &[u32]) -> u32 {
    instructions(words)
        .find(|(op, _)| *op == Op::EntryPoint)
        .unwrap()
        .1[0]
}

// SPIR-V fields are from the Khronos grammar's OpDecorate (:838-846),
// OpVariable (:679-689), OpTypePointer (:433-441), OpTypeImage (:354-369).
fn descriptors(words: &[u32]) -> BTreeMap<(u32, u32), vk::DescriptorType> {
    let mut sets = BTreeMap::new();
    let mut bindings = BTreeMap::new();
    let mut types = BTreeMap::new();
    let mut pointers = BTreeMap::new();
    for (op, a) in instructions(words) {
        match op {
            Op::Decorate if a[1] == 34 => {
                sets.insert(a[0], a[2]);
            }
            Op::Decorate if a[1] == 33 => {
                bindings.insert(a[0], a[2]);
            }
            Op::TypeImage => {
                types.insert(
                    a[0],
                    if a[6] == 2 {
                        vk::DescriptorType::STORAGE_IMAGE
                    } else {
                        vk::DescriptorType::SAMPLED_IMAGE
                    },
                );
            }
            Op::TypeSampler => {
                types.insert(a[0], vk::DescriptorType::SAMPLER);
            }
            Op::TypeSampledImage => {
                types.insert(a[0], vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
            }
            Op::TypePointer => {
                pointers.insert(a[0], a[2]);
            }
            _ => {}
        }
    }
    let mut result = BTreeMap::new();
    for (op, a) in instructions(words) {
        if op != Op::Variable {
            continue;
        }
        let (Some(&set), Some(&binding)) = (sets.get(&a[1]), bindings.get(&a[1])) else {
            continue;
        };
        let ty = match a[2] {
            0 => types[&pointers[&a[0]]],
            2 => vk::DescriptorType::UNIFORM_BUFFER,
            12 => vk::DescriptorType::STORAGE_BUFFER,
            storage => panic!("unexpected descriptor storage {storage}"),
        };
        result.insert((set, binding), ty);
    }
    result
}

#[test]
#[ignore = "cold RADV pipeline compilation benchmark"]
fn cold_pipeline_compile_times() {
    assert_eq!(std::env::var("RADV_DEBUG").as_deref(), Ok("nocache"));
    assert_eq!(
        std::env::var("MESA_SHADER_CACHE_DISABLE").as_deref(),
        Ok("true")
    );
    let dir = std::env::temp_dir().join("shadowbox-spv-opt-measurements");
    let mut paths = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".raw.spv"))
        .collect::<Vec<_>>();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "run export_optimization_measurements first"
    );
    let rounds =
        std::env::var("SHADOWBOX_PIPELINE_ROUNDS").map_or(9, |s| s.parse::<usize>().unwrap());
    assert!(rounds >= 3);
    let mut rows = "name\traw_pipeline_us\tours_pipeline_us\toffline_pipeline_us\n".to_owned();
    let mut sample_rows = "name\tround\tvariant\tpipeline_us\n".to_owned();
    unsafe {
        let entry = Entry::load().unwrap();
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(
                    &vk::ApplicationInfo::default()
                        .application_name(c"shadowbox-spv-bench")
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
            .expect("RADV hardware is required for this benchmark");
        let props = instance.get_physical_device_properties(physical);
        let device_info = format!(
            "device={} driver_version={} rounds={}\n",
            props.device_name_as_c_str().unwrap().to_string_lossy(),
            props.driver_version,
            rounds
        );
        println!("{device_info}");
        fs::write(dir.join("device.txt"), device_info).unwrap();
        let family = instance
            .get_physical_device_queue_family_properties(physical)
            .iter()
            .position(|p| {
                p.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .unwrap() as u32;
        // Enable optional image features used by corpus capabilities, keeping
        // buffer robustness at the original harness's default. These fields
        // are defined in the public Vulkan header, vulkan_core.h:3526-3559.
        let supported = instance.get_physical_device_features(physical);
        let features = vk::PhysicalDeviceFeatures {
            image_cube_array: supported.image_cube_array,
            shader_image_gather_extended: supported.shader_image_gather_extended,
            shader_storage_image_read_without_format: supported
                .shader_storage_image_read_without_format,
            shader_storage_image_write_without_format: supported
                .shader_storage_image_write_without_format,
            ..Default::default()
        };
        let device = instance
            .create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .enabled_features(&features)
                    .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(family)
                        .queue_priorities(&[1.0])]),
                None,
            )
            .unwrap();
        for path in paths {
            let name = path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .strip_suffix(".raw.spv")
                .unwrap();
            let variants = ["raw", "ours", "offline"];
            let binaries = variants.map(|v| words(&dir.join(format!("{name}.{v}.spv"))));
            let stage_model = model(&binaries[0]);
            let companion = match stage_model {
                4 => Some(words(&dir.join(format!("{name}.vert.spv")))),
                0 => Some(words(&dir.join(format!("{name}.frag.spv")))),
                5 => None,
                other => panic!("unexpected execution model {other}"),
            };
            let shaders = binaries.each_ref().map(|w| {
                device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(w), None)
                    .unwrap()
            });
            let companion_shader = companion.as_ref().map(|w| {
                device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(w), None)
                    .unwrap()
            });
            let mut descriptor_map = descriptors(&binaries[0]);
            if let Some(w) = &companion {
                descriptor_map.extend(descriptors(w));
            }
            let mut set_layouts = Vec::new();
            if let Some(max_set) = descriptor_map.keys().map(|(set, _)| *set).max() {
                for set in 0..=max_set {
                    let bindings = descriptor_map
                        .iter()
                        .filter(|((s, _), _)| *s == set)
                        .map(|((_, binding), ty)| {
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(*binding)
                                .descriptor_type(*ty)
                                .descriptor_count(1)
                                .stage_flags(vk::ShaderStageFlags::ALL)
                        })
                        .collect::<Vec<_>>();
                    set_layouts.push(
                        device
                            .create_descriptor_set_layout(
                                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                                None,
                            )
                            .unwrap(),
                    );
                }
            }
            let layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
                .unwrap();
            let render = if stage_model == 5 {
                None
            } else {
                let frag = if stage_model == 4 {
                    &binaries[0]
                } else {
                    companion.as_ref().unwrap()
                };
                let locations = instructions(frag)
                    .filter(|(op, a)| *op == Op::Decorate && a[1] == 30)
                    .map(|(_, a)| a[2])
                    .max()
                    .unwrap_or(0)
                    .min(7)
                    + 1;
                let attachments = (0..locations)
                    .map(|_| {
                        vk::AttachmentDescription::default()
                            .format(vk::Format::R32G32B32A32_SFLOAT)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .initial_layout(vk::ImageLayout::UNDEFINED)
                            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    })
                    .collect::<Vec<_>>();
                let references = (0..locations)
                    .map(|i| vk::AttachmentReference {
                        attachment: i,
                        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    })
                    .collect::<Vec<_>>();
                Some((
                    device
                        .create_render_pass(
                            &vk::RenderPassCreateInfo::default()
                                .attachments(&attachments)
                                .subpasses(&[vk::SubpassDescription::default()
                                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                                    .color_attachments(&references)]),
                            None,
                        )
                        .unwrap(),
                    locations,
                ))
            };
            let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
            for round in 0..rounds {
                for offset in 0..3 {
                    let variant = (round + offset) % 3;
                    let shader = shaders[variant];
                    let stage = |flags, module| {
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(flags)
                            .module(module)
                            .name(c"main")
                    };
                    let (pipeline, elapsed) = if stage_model == 5 {
                        let info = vk::ComputePipelineCreateInfo::default()
                            .stage(stage(vk::ShaderStageFlags::COMPUTE, shader))
                            .layout(layout);
                        let start = Instant::now();
                        let pipeline = device
                            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                            .unwrap()[0];
                        (pipeline, start.elapsed())
                    } else {
                        let stages = [
                            stage(
                                vk::ShaderStageFlags::VERTEX,
                                if stage_model == 0 {
                                    shader
                                } else {
                                    companion_shader.unwrap()
                                },
                            ),
                            stage(
                                vk::ShaderStageFlags::FRAGMENT,
                                if stage_model == 4 {
                                    shader
                                } else {
                                    companion_shader.unwrap()
                                },
                            ),
                        ];
                        // Corpus vertex inputs use locations 0..31. Pipeline
                        // creation does not read buffers. Each location has a
                        // vec4 float format, as in the existing Vulkan harness.
                        let attributes = (0..32)
                            .map(|location| vk::VertexInputAttributeDescription {
                                location,
                                binding: 0,
                                format: vk::Format::R32G32B32A32_SFLOAT,
                                offset: location * 16,
                            })
                            .collect::<Vec<_>>();
                        let binding = [vk::VertexInputBindingDescription {
                            binding: 0,
                            stride: 512,
                            input_rate: vk::VertexInputRate::VERTEX,
                        }];
                        let vertex = vk::PipelineVertexInputStateCreateInfo::default()
                            .vertex_attribute_descriptions(&attributes)
                            .vertex_binding_descriptions(&binding);
                        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
                        let viewports = [vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: 32.0,
                            height: 32.0,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        }];
                        let scissors = [vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent: vk::Extent2D {
                                width: 32,
                                height: 32,
                            },
                        }];
                        let viewport = vk::PipelineViewportStateCreateInfo::default()
                            .viewports(&viewports)
                            .scissors(&scissors);
                        let raster = vk::PipelineRasterizationStateCreateInfo::default()
                            .polygon_mode(vk::PolygonMode::FILL)
                            .cull_mode(vk::CullModeFlags::NONE)
                            .line_width(1.0);
                        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
                        let colors = (0..render.unwrap().1)
                            .map(|_| {
                                vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                            })
                            .collect::<Vec<_>>();
                        let blend =
                            vk::PipelineColorBlendStateCreateInfo::default().attachments(&colors);
                        let depth = vk::PipelineDepthStencilStateCreateInfo::default();
                        let info = vk::GraphicsPipelineCreateInfo::default()
                            .stages(&stages)
                            .vertex_input_state(&vertex)
                            .input_assembly_state(&assembly)
                            .viewport_state(&viewport)
                            .rasterization_state(&raster)
                            .multisample_state(&multisample)
                            .color_blend_state(&blend)
                            .depth_stencil_state(&depth)
                            .layout(layout)
                            .render_pass(render.unwrap().0);
                        let start = Instant::now();
                        let pipeline = device
                            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
                            .unwrap()[0];
                        (pipeline, start.elapsed())
                    };
                    let us = elapsed.as_secs_f64() * 1e6;
                    samples[variant].push(us);
                    sample_rows.push_str(&format!(
                        "{name}\t{round}\t{}\t{us:.3}\n",
                        variants[variant]
                    ));
                    device.destroy_pipeline(pipeline, None);
                }
            }
            let medians = samples.map(|mut s| {
                s.sort_by(f64::total_cmp);
                s[s.len() / 2]
            });
            rows.push_str(&format!(
                "{name}\t{:.3}\t{:.3}\t{:.3}\n",
                medians[0], medians[1], medians[2]
            ));
            println!(
                "{name}: raw={:.1} ours={:.1} offline={:.1} us",
                medians[0], medians[1], medians[2]
            );
            for shader in shaders {
                device.destroy_shader_module(shader, None);
            }
            if let Some(shader) = companion_shader {
                device.destroy_shader_module(shader, None);
            }
            if let Some((render, _)) = render {
                device.destroy_render_pass(render, None);
            }
            device.destroy_pipeline_layout(layout, None);
            for set in set_layouts {
                device.destroy_descriptor_set_layout(set, None);
            }
        }
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    fs::write(dir.join("pipelines.tsv"), rows).unwrap();
    fs::write(dir.join("pipeline-samples.tsv"), sample_rows).unwrap();
}
