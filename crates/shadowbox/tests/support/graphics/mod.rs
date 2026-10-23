//! Minimal Vulkan graphics transport for exact stage comparisons.
//!
//! The tested SPIR-V is installed unchanged.  Observation is limited to a
//! caller-provided storage buffer and ordinary color/depth attachments.
// Vulkan API field layouts: /usr/include/vulkan/vulkan_core.h:3898-3920
// ImageCreateInfo,3996-4004 BufferImageCopy,4038-4043 MemoryBarrier,
// 4207-4214 DescriptorSetLayoutBinding,4267-4273 vertex attributes,
// 4311-4326 depth/stencil,4354-4368 rasterization,4370-4376 tessellation,
// 4397-4417 graphics pipeline,4419-4431 attachments,4520-4532 render pass.
// Stage flag values: vulkan_core.h:2882-2888. The fixture shaders, formats,
// clear values and capture schema are project-owned test transport choices.

#[path = "../shader_features.rs"]
mod shader_features;
#[path = "../texture_sampled.rs"]
mod texture;
use ash::{vk, Entry};
use shadowbox::headers::Stage;
use shadowbox::spirv::Op;
use std::{mem::size_of, ptr};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const STORAGE_WORDS: usize = 4096;
const SENTINEL: u32 = 0xdead_beef;

pub struct Draw<'a> {
    pub shaders: &'a [(Stage, Vec<u32>)],
    pub vertices: &'a [[f32; 8]],
    pub vertex_count: u32,
    pub topology: vk::PrimitiveTopology,
    pub discard: bool,
    pub samples: vk::SampleCountFlags,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Observation {
    pub storage: Vec<u32>,
    pub color: Vec<u32>,
    pub depth: Vec<u32>,
}

struct Buffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    bytes: vk::DeviceSize,
}

struct Image {
    handle: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

pub struct Graphics {
    _entry: Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    shader_features: shader_features::ShaderFeatures,
    subgroup32_stages: vk::ShaderStageFlags,
    extent: vk::Extent2D,
    timestamp_bits: u32,
    pub description: String,
}

impl Graphics {
    pub fn new() -> Self {
        unsafe {
            let entry = Entry::load().unwrap();
            let app_name = c"shadowbox-stage-exact";
            let app = vk::ApplicationInfo::default()
                .application_name(app_name)
                .api_version(vk::API_VERSION_1_3);
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                )
                .unwrap();
            let physical = instance
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .find(|physical| {
                    instance
                        .get_physical_device_properties(*physical)
                        .device_name_as_c_str()
                        .unwrap()
                        .to_string_lossy()
                        .contains("RADV")
                })
                .expect("graphics exact checks require RADV hardware; no silent skip");
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
                .position(|properties| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .expect("RADV device has no graphics queue") as u32;
            let timestamp_bits = instance.get_physical_device_queue_family_properties(physical)
                [family as usize]
                .timestamp_valid_bits;
            let priorities = [1.0];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities)];
            // /usr/include/vulkan/vulkan_core.h:3531-3532,3552-3553,3564:
            // geometry/tessellation, stage storage writes, and ClipDistance.
            let features = vk::PhysicalDeviceFeatures::default()
                .geometry_shader(true)
                .tessellation_shader(true)
                .vertex_pipeline_stores_and_atomics(true)
                .fragment_stores_and_atomics(true)
                .shader_clip_distance(true);
            let shader_features = shader_features::ShaderFeatures::query(&instance, physical);
            let description = format!("{description} {}", shader_features.describe());
            let extensions = shader_features.extension_names();
            // VOTE requires size 32 for both translated and reference shaders.
            // Vulkan 1.3 core fields, vulkan_core.h:7771-7791.
            let mut subgroup = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
            let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut subgroup);
            instance.get_physical_device_features2(physical, &mut query);
            let mut subgroup_properties =
                vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut query =
                vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_properties);
            instance.get_physical_device_properties2(physical, &mut query);
            let subgroup32_stages = if subgroup.subgroup_size_control != 0
                && subgroup_properties.min_subgroup_size <= 32
                && subgroup_properties.max_subgroup_size >= 32
            {
                subgroup_properties.required_subgroup_size_stages
            } else {
                vk::ShaderStageFlags::empty()
            };
            // Only subgroupSizeControl is used. computeFullSubgroups stays disabled.
            subgroup.compute_full_subgroups = vk::FALSE;
            let description =
                format!("{description} required_subgroup_size_32_stages={subgroup32_stages:?}");
            let mut reconvergence =
                vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default()
                    .shader_maximal_reconvergence(shader_features.maximal_reconvergence);
            let mut quad = vk::PhysicalDeviceShaderQuadControlFeaturesKHR::default()
                .shader_quad_control(shader_features.quad_control);
            let mut demote = vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures::default()
                .shader_demote_to_helper_invocation(shader_features.demote_to_helper_invocation);
            let mut device_info = vk::DeviceCreateInfo::default().push_next(&mut subgroup);
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
                        .queue_create_infos(&queue_info)
                        .enabled_features(&features),
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
            Self {
                _entry: entry,
                instance,
                physical,
                device,
                queue,
                pool,
                command,
                description,
                shader_features,
                subgroup32_stages,
                extent: vk::Extent2D {
                    width: WIDTH,
                    height: HEIGHT,
                },
                timestamp_bits,
            }
        }
    }

    /// Test transport for one immutable 2D image: set1 image0/sampler1.
    /// Vulkan layouts: vulkan_core.h:4179-4197,4207-4214,4223-4234.
    pub fn run_with_texture(&mut self, draw: &Draw<'_>, sampled: bool) -> Observation {
        self.run_draw(draw, sampled, None).0
    }

    pub fn run_timed(
        &mut self,
        draw: &Draw<'_>,
        instances: u32,
        rounds: usize,
    ) -> (Observation, Vec<f64>) {
        assert!(instances > 0 && rounds > 0 && self.timestamp_bits > 0);
        let saved = self.extent;
        self.extent = vk::Extent2D {
            width: 256,
            height: 256,
        };
        let result = self.run_draw(draw, false, Some((instances, rounds)));
        self.extent = saved;
        result
    }

    fn run_draw(
        &mut self,
        draw: &Draw<'_>,
        sampled: bool,
        timing: Option<(u32, usize)>,
    ) -> (Observation, Vec<f64>) {
        assert!(matches!(
            draw.samples,
            vk::SampleCountFlags::TYPE_1 | vk::SampleCountFlags::TYPE_4
        ));
        let depth_words = self.extent.width as usize * self.extent.height as usize;
        let color_words = depth_words * 4;
        assert!(!draw.shaders.is_empty());
        for (_, words) in draw.shaders {
            self.shader_features.require_module(words);
            assert!(
                descriptor_sets(words)
                    .iter()
                    .all(|set| *set == 3 || (sampled && *set == 1)),
                "graphics exact transport binds only storage set3 and optional sampled image set1"
            );
        }

        unsafe {
            // Public timestamp query API, vulkan_core.h:3677,3723,3878-3885,5063-5067.
            // Query allocation, pipeline construction and readback are outside the interval.
            let query_pool = timing.map(|_| {
                self.device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                    .unwrap()
            });
            let properties = self.instance.get_physical_device_properties(self.physical);
            assert!(timing.is_none() || properties.limits.timestamp_compute_and_graphics != 0);
            let storage = self.create_buffer(
                (STORAGE_WORDS * size_of::<u32>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            );
            ptr::write_bytes(storage.mapped.cast::<u32>(), 0, STORAGE_WORDS);
            for word in std::slice::from_raw_parts_mut(storage.mapped.cast::<u32>(), STORAGE_WORDS)
            {
                *word = SENTINEL;
            }

            let vertex_slots = draw.vertices.len().max(draw.vertex_count as usize).max(1);
            let vertex_bytes = vertex_slots * size_of::<[f32; 128]>();
            let vertices =
                self.create_buffer(vertex_bytes as u64, vk::BufferUsageFlags::VERTEX_BUFFER);
            ptr::write_bytes(vertices.mapped, 0, vertex_bytes);
            // Project-owned data: locations 0/1 retain position/payload;
            // all other locations have distinct finite binary fractions.
            // Both reference and tested pipelines receive these exact bytes.
            for (vertex, input) in draw.vertices.iter().enumerate() {
                for location in 0..32 {
                    let mut value: [f32; 4] = if location == 0 {
                        input[..4].try_into().unwrap()
                    } else {
                        input[4..].try_into().unwrap()
                    };
                    if location > 1 {
                        value[0] += location as f32 * 0.125;
                        value[1] -= location as f32 * 0.25;
                        value[3] += location as f32;
                    }
                    ptr::copy_nonoverlapping(
                        value.as_ptr(),
                        vertices
                            .mapped
                            .cast::<f32>()
                            .add(vertex * 128 + location * 4),
                        4,
                    );
                }
            }

            let readback_words = color_words
                + if draw.samples == vk::SampleCountFlags::TYPE_1 {
                    depth_words
                } else {
                    0
                };
            let readback = self.create_buffer(
                (readback_words * size_of::<u32>()) as u64,
                vk::BufferUsageFlags::TRANSFER_DST,
            );
            ptr::write_bytes(readback.mapped, 0, readback.bytes as usize);

            let color = self.create_image(
                vk::Format::R32G32B32A32_SFLOAT,
                draw.samples,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | if draw.samples == vk::SampleCountFlags::TYPE_1 {
                        vk::ImageUsageFlags::TRANSFER_SRC
                    } else {
                        vk::ImageUsageFlags::empty()
                    },
                vk::ImageAspectFlags::COLOR,
            );
            let resolve = (draw.samples == vk::SampleCountFlags::TYPE_4).then(|| {
                self.create_image(
                    vk::Format::R32G32B32A32_SFLOAT,
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                    vk::ImageAspectFlags::COLOR,
                )
            });
            // Multisample images cannot be copied directly to a buffer.  The
            // initial 4x path observes resolved coverage and intentionally has
            // no depth attachment or per-sample identity.
            let depth = (draw.samples == vk::SampleCountFlags::TYPE_1).then(|| {
                self.create_image(
                    vk::Format::D32_SFLOAT,
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                    vk::ImageAspectFlags::DEPTH,
                )
            });

            let texture = sampled.then(|| {
                texture::Texture::new(
                    &self.instance,
                    &self.device,
                    self.physical,
                    self.command,
                    self.queue,
                    (1, false, false),
                )
            });
            let sampled_bindings = if sampled {
                vec![
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                ]
            } else {
                Vec::new()
            };
            let descriptor_binding = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)];
            let descriptor_layouts = (0..4)
                .map(|set| {
                    self.device
                        .create_descriptor_set_layout(
                            &vk::DescriptorSetLayoutCreateInfo::default().bindings(if set == 3 {
                                &descriptor_binding
                            } else if set == 1 {
                                &sampled_bindings
                            } else {
                                &[]
                            }),
                            None,
                        )
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let pipeline_layout = self
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_layouts),
                    None,
                )
                .unwrap();
            let descriptor_pool = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(2)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_BUFFER,
                                descriptor_count: 1,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::SAMPLED_IMAGE,
                                descriptor_count: 1,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::SAMPLER,
                                descriptor_count: 1,
                            },
                        ]),
                    None,
                )
                .unwrap();
            let descriptor_set = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&descriptor_layouts[3..]),
                )
                .unwrap()[0];
            let storage_info = [vk::DescriptorBufferInfo::default()
                .buffer(storage.handle)
                .range(storage.bytes)];
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&storage_info)],
                &[],
            );

            let texture_set = texture.as_ref().map(|texture| {
                let set = self
                    .device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pool)
                            .set_layouts(&descriptor_layouts[1..2]),
                    )
                    .unwrap()[0];
                let image = [vk::DescriptorImageInfo::default()
                    .image_view(texture.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let sampler = [vk::DescriptorImageInfo::default().sampler(texture.sampler)];
                self.device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(&image),
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(&sampler),
                    ],
                    &[],
                );
                set
            });

            let (render_pass, framebuffer) =
                self.create_render_targets(draw.samples, &color, resolve.as_ref(), depth.as_ref());
            let modules = draw
                .shaders
                .iter()
                .map(|(stage, words)| {
                    let module = self
                        .device
                        .create_shader_module(
                            &vk::ShaderModuleCreateInfo::default().code(words),
                            None,
                        )
                        .unwrap();
                    (*stage, module)
                })
                .collect::<Vec<_>>();
            let entry_name = c"main";
            let mut required_sizes = vec![
                    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                        .required_subgroup_size(32);
                    modules.len()
                ];
            let stages = modules
                .iter()
                .zip(draw.shaders)
                .zip(&mut required_sizes)
                .map(|(((stage, module), (_, words)), required_size)| {
                    let mut info = vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk_stage(*stage))
                        .module(*module)
                        .name(entry_name);
                    // Vote/Ballot capabilities, Mesa spirv.h:1134,1136.
                    // Inspect actual modules so GLSL and raw SPIR-V also get 32.
                    let mut at = 5;
                    let mut requires32 = false;
                    while at < words.len() {
                        if Op::of(words[at]) == Op::Capability && matches!(words[at + 1], 62 | 64) {
                            requires32 = true;
                            break;
                        }
                        at += (words[at] >> 16) as usize;
                    }
                    if requires32 {
                        assert!(
                            self.subgroup32_stages.contains(vk_stage(*stage)),
                            "device cannot bind required subgroup size 32 for {stage:?}"
                        );
                        info = info.push_next(required_size);
                    }
                    info
                })
                .collect::<Vec<_>>();
            let vertex_binding = [vk::VertexInputBindingDescription {
                binding: 0,
                stride: 512,
                input_rate: vk::VertexInputRate::VERTEX,
            }];
            // The dynamic input array declares all 32 Maxwell locations.
            // Vulkan requires descriptions for consumed vertex attributes.
            // /usr/include/vulkan/vulkan_core.h:4267-4273.
            let vertex_attributes = (0..32)
                .map(|location| vk::VertexInputAttributeDescription {
                    location,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: location * 16,
                })
                .collect::<Vec<_>>();
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_binding)
                .vertex_attribute_descriptions(&vertex_attributes);
            let input_assembly =
                vk::PipelineInputAssemblyStateCreateInfo::default().topology(draw.topology);
            let tessellation =
                vk::PipelineTessellationStateCreateInfo::default().patch_control_points(3);
            let viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.extent.width as f32,
                height: self.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.extent.width,
                    height: self.extent.height,
                },
            }];
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewports(&viewport)
                .scissors(&scissors);
            let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
                .rasterizer_discard_enable(draw.discard)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(draw.samples);
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::ALWAYS)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0);
            let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend =
                vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
            let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .tessellation_state(&tessellation)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization)
                .multisample_state(&multisample)
                .depth_stencil_state(&depth_stencil)
                .color_blend_state(&blend)
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0)];
            let pipeline = self
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                .unwrap()[0];

            self.device
                .reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .unwrap();
            self.device
                .begin_command_buffer(self.command, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            if let Some(pool) = query_pool {
                self.device.cmd_reset_query_pool(self.command, pool, 0, 2);
            }
            let clear_values = if depth.is_some() {
                vec![
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [-16.0, -32.0, -64.0, -128.0],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ]
            } else {
                vec![
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [-16.0, -32.0, -64.0, -128.0],
                        },
                    },
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [-16.0, -32.0, -64.0, -128.0],
                        },
                    },
                ]
            };
            self.device.cmd_begin_render_pass(
                self.command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: self.extent.width,
                            height: self.extent.height,
                        },
                    })
                    .clear_values(&clear_values),
                vk::SubpassContents::INLINE,
            );
            self.device
                .cmd_bind_pipeline(self.command, vk::PipelineBindPoint::GRAPHICS, pipeline);
            self.device.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                3,
                &[descriptor_set],
                &[],
            );
            if let Some(set) = texture_set {
                self.device.cmd_bind_descriptor_sets(
                    self.command,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    1,
                    &[set],
                    &[],
                );
            }
            self.device
                .cmd_bind_vertex_buffers(self.command, 0, &[vertices.handle], &[0]);
            if let Some(pool) = query_pool {
                self.device.cmd_write_timestamp(
                    self.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    pool,
                    0,
                );
            }
            self.device.cmd_draw(
                self.command,
                draw.vertex_count,
                timing.map_or(1, |t| t.0),
                0,
                0,
            );
            if let Some(pool) = query_pool {
                self.device.cmd_write_timestamp(
                    self.command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    1,
                );
            }
            self.device.cmd_end_render_pass(self.command);
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
                &[],
                &[],
            );
            let color_source = resolve.as_ref().unwrap_or(&color);
            self.copy_image(
                color_source.handle,
                vk::ImageAspectFlags::COLOR,
                readback.handle,
                0,
            );
            if let Some(depth) = &depth {
                self.copy_image(
                    depth.handle,
                    vk::ImageAspectFlags::DEPTH,
                    readback.handle,
                    (color_words * size_of::<u32>()) as u64,
                );
            }
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::TRANSFER,
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
            self.device.end_command_buffer(self.command).unwrap();
            let mut times = Vec::new();
            for _ in 0..timing.map_or(1, |t| t.1) {
                self.submit_and_wait();
                if let Some(pool) = query_pool {
                    let mut ticks = [0u64; 2];
                    self.device
                        .get_query_pool_results(
                            pool,
                            0,
                            &mut ticks,
                            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                        )
                        .unwrap();
                    let mask = u64::MAX >> (64 - self.timestamp_bits);
                    times.push(
                        (ticks[1].wrapping_sub(ticks[0]) & mask) as f64
                            * f64::from(properties.limits.timestamp_period)
                            / 1000.0,
                    );
                }
            }
            if let Some(pool) = query_pool {
                self.device.destroy_query_pool(pool, None);
            }

            let storage_words =
                std::slice::from_raw_parts(storage.mapped.cast::<u32>(), STORAGE_WORDS).to_vec();
            let readback_words =
                std::slice::from_raw_parts(readback.mapped.cast::<u32>(), readback_words);
            let observation = Observation {
                storage: storage_words,
                color: readback_words[..color_words].to_vec(),
                depth: if depth.is_some() {
                    readback_words[color_words..].to_vec()
                } else {
                    Vec::new()
                },
            };

            self.device.destroy_pipeline(pipeline, None);
            for (_, module) in modules {
                self.device.destroy_shader_module(module, None);
            }
            self.device.destroy_framebuffer(framebuffer, None);
            self.device.destroy_render_pass(render_pass, None);
            self.device.destroy_descriptor_pool(descriptor_pool, None);
            self.device.destroy_pipeline_layout(pipeline_layout, None);
            for layout in descriptor_layouts {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            if let Some(image) = depth {
                self.destroy_image(image);
            }
            if let Some(image) = resolve {
                self.destroy_image(image);
            }
            self.destroy_image(color);
            if let Some(texture) = texture {
                texture.destroy(&self.device);
            }
            self.destroy_buffer(readback);
            self.destroy_buffer(vertices);
            self.destroy_buffer(storage);
            (observation, times)
        }
    }

    unsafe fn create_buffer(&self, bytes: vk::DeviceSize, usage: vk::BufferUsageFlags) -> Buffer {
        let handle = self
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default().size(bytes).usage(usage),
                None,
            )
            .unwrap();
        let requirements = self.device.get_buffer_memory_requirements(handle);
        let memory = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(self.memory_type(
                        requirements.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )),
                None,
            )
            .unwrap();
        self.device.bind_buffer_memory(handle, memory, 0).unwrap();
        let mapped = self
            .device
            .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())
            .unwrap()
            .cast();
        Buffer {
            handle,
            memory,
            mapped,
            bytes,
        }
    }

    unsafe fn create_image(
        &self,
        format: vk::Format,
        samples: vk::SampleCountFlags,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Image {
        let handle = self
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: self.extent.width,
                        height: self.extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(samples)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .unwrap();
        let requirements = self.device.get_image_memory_requirements(handle);
        let memory = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(self.memory_type(
                        requirements.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )),
                None,
            )
            .unwrap();
        self.device.bind_image_memory(handle, memory, 0).unwrap();
        let view = self
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(handle)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
            .unwrap();
        Image {
            handle,
            memory,
            view,
        }
    }

    unsafe fn create_render_targets(
        &self,
        samples: vk::SampleCountFlags,
        color: &Image,
        resolve: Option<&Image>,
        depth: Option<&Image>,
    ) -> (vk::RenderPass, vk::Framebuffer) {
        let color_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let mut attachments = vec![vk::AttachmentDescription::default()
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(if resolve.is_some() {
                vk::AttachmentStoreOp::DONT_CARE
            } else {
                vk::AttachmentStoreOp::STORE
            })
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(if resolve.is_some() {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            } else {
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            })];
        let mut views = vec![color.view];
        let resolve_ref;
        let depth_ref;
        let mut subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_ref);
        if let Some(resolve) = resolve {
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            );
            views.push(resolve.view);
            resolve_ref = [vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            subpass = subpass.resolve_attachments(&resolve_ref);
        }
        if let Some(depth) = depth {
            let index = attachments.len() as u32;
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(vk::Format::D32_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            );
            views.push(depth.view);
            depth_ref = vk::AttachmentReference {
                attachment: index,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };
            subpass = subpass.depth_stencil_attachment(&depth_ref);
        }
        let subpasses = [subpass];
        let render_pass = self
            .device
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses),
                None,
            )
            .unwrap();
        let framebuffer = self
            .device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&views)
                    .width(self.extent.width)
                    .height(self.extent.height)
                    .layers(1),
                None,
            )
            .unwrap();
        (render_pass, framebuffer)
    }

    unsafe fn copy_image(
        &self,
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
    ) {
        self.device.cmd_copy_image_to_buffer(
            self.command,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: aspect,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: self.extent.width,
                    height: self.extent.height,
                    depth: 1,
                })],
        );
    }

    unsafe fn submit_and_wait(&self) {
        submit_and_wait(&self.device, self.queue, self.command);
    }

    unsafe fn destroy_buffer(&self, buffer: Buffer) {
        self.device.unmap_memory(buffer.memory);
        self.device.destroy_buffer(buffer.handle, None);
        self.device.free_memory(buffer.memory, None);
    }

    unsafe fn destroy_image(&self, image: Image) {
        self.device.destroy_image_view(image.view, None);
        self.device.destroy_image(image.handle, None);
        self.device.free_memory(image.memory, None);
    }

    unsafe fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> u32 {
        let properties = self
            .instance
            .get_physical_device_memory_properties(self.physical);
        (0..properties.memory_type_count)
            .find(|index| {
                bits & (1 << index) != 0
                    && properties.memory_types[*index as usize]
                        .property_flags
                        .contains(flags)
            })
            .expect("RADV has no compatible memory type")
    }
}

impl Drop for Graphics {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn vk_stage(stage: Stage) -> vk::ShaderStageFlags {
    match stage {
        Stage::Vertex => vk::ShaderStageFlags::VERTEX,
        Stage::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        Stage::TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        Stage::Geometry => vk::ShaderStageFlags::GEOMETRY,
        Stage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        Stage::Compute => panic!("compute module supplied to Graphics"),
    }
}

fn descriptor_sets(words: &[u32]) -> Vec<u32> {
    // SPIR-V OpDecorate has opcode 71; decoration 34 is DescriptorSet.
    let mut sets = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let word_count = (words[at] >> 16) as usize;
        let opcode = Op::of(words[at]);
        assert!(
            word_count != 0 && at + word_count <= words.len(),
            "malformed SPIR-V"
        );
        if opcode == Op::Decorate && word_count >= 4 && words[at + 2] == 34 {
            sets.push(words[at + 3]);
        }
        at += word_count;
    }
    sets
}

// Bounded submission also used for the shared immutable image upload.
fn submit_and_wait(device: &ash::Device, queue: vk::Queue, command: vk::CommandBuffer) {
    unsafe {
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        let command_buffers = [command];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        let result = device
            .queue_submit(queue, &submits, fence)
            .and_then(|()| device.wait_for_fences(&[fence], true, 5_000_000_000));
        if let Err(error) = result {
            eprintln!("graphics GPU submission failed: {error:?}; fence deadline=5 seconds");
            std::process::exit(124);
        }
        device.destroy_fence(fence, None);
    }
}
