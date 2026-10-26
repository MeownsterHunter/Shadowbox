//! Cache behavior with project-authored programs. No instruction mapping changes.
use shadowbox::{
    disk_cache_stats, emit_translation_cache,
    headers::Stage,
    interface::{
        BoundTexture, GeometryInput, GraphicsState, PatchIo, PatchLayout, ShaderInput,
        TessellationDomain, TessellationSpacing, TessellationState, TextureDimension, Translation,
    },
    merge_disk_caches, translate, translate_cached_with_context,
    translate_with_options_and_graphics_state, Cache, CacheConfig, CacheContext, TargetGeneration,
};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

/// A program's identity under `context`, through [`shadowbox::Program`].
fn identity(
    input: &shadowbox::interface::ShaderInput<'_>,
    context: shadowbox::CacheContext,
) -> Result<String, shadowbox::interface::Stopped> {
    Ok(shadowbox::Program::parse(input, context.graphics_state)?
        .identity(context.generation, context.options))
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir()
            .join("shadowbox-translation-cache-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn header(stage: u32) -> Vec<u8> {
    // Mesa cla097sph.h:37-42,66-73 and sph.rs:204-215,305-350:
    // stage10..14, threads88..96, topology120..124, vertices128..140.
    let mut words = [0u32; 32];
    words[0] = stage << 10;
    words[2] = 1 << 24;
    words[3] = 1 << 24;
    words[4] = 1;
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}
fn code(include_vote: bool) -> Vec<u8> {
    // Mesa sm50.rs:3418-3428 bundle; :243-244 absent barriers;
    // :3106-3110 EXIT/CC.T; :3183-3187 NOP; :220-235 PT;
    // :3227-3247 VOTE all, ballot0, predicate destination45..48=PT,
    // predicate source39..42=PT, no negation42 and mode48..50=all.
    let nop = 0x50b0_0000_0007_0f00u64;
    let vote = (0x50d8u64 << 48) | (7 << 16) | (7 << 39) | (7 << 45);
    let control = 0x7e0u64 | (0x7e0 << 21) | (0x7e0 << 42);
    [
        control,
        if include_vote { vote } else { nop },
        0xe300_0000_0007_000f,
        nop,
    ]
    .into_iter()
    .flat_map(u64::to_le_bytes)
    .collect()
}
fn cache(dir: &Path, memory: shadowbox::cache::Keep) -> Cache {
    Cache::new(CacheConfig {
        memory,
        disk: shadowbox::cache::Disk::at(dir),
    })
}

#[test]
fn hits_preserve_metadata_without_invoking_translation_and_corruption_retranslates() {
    let temp = Temp::new();
    let code = code(true);
    let header = header(0);
    let input = ShaderInput {
        code: &code,
        program_record: &header,
    };
    let cold = Result::from(translate(shadowbox::QuietRun, &input)).unwrap();
    assert!(cold.requires_subgroup_size_32);
    let mut first = cache(&temp.0, shadowbox::cache::Keep::outputs());
    let output = Result::from(translate_cached_with_context(
        shadowbox::QuietRun,
        &input,
        CacheContext::default(),
        &mut first,
    ))
    .unwrap();
    assert_eq!(first.stats.misses, 1);
    assert_eq!(output.spirv, cold.spirv);
    let hit =
        Result::from(first.translate(&input, Stage::Compute, || panic!("memory hit translated")))
            .unwrap();
    assert_eq!(first.stats.memory_hits, 1);
    assert_eq!(hit.spirv, cold.spirv);
    assert!(hit.requires_subgroup_size_32);
    let mut disk = cache(&temp.0, shadowbox::cache::Keep::nothing());
    let hit =
        Result::from(disk.translate(&input, Stage::Compute, || panic!("disk hit translated")))
            .unwrap();
    assert_eq!(disk.stats.disk_hits, 1);
    assert!(hit.requires_subgroup_size_32);
    assert_eq!(disk_cache_stats(&temp.0).unwrap().subgroup_size_32, 1);
    let key = identity(&input, CacheContext::default()).unwrap();
    fs::write(temp.0.join(format!("{key}.sbc")), b"truncated").unwrap();
    assert_eq!(disk_cache_stats(&temp.0).unwrap().invalid, 1);
    let mut repaired = cache(&temp.0, shadowbox::cache::Keep::nothing());
    let output = Result::from(translate_cached_with_context(
        shadowbox::QuietRun,
        &input,
        CacheContext::default(),
        &mut repaired,
    ))
    .unwrap();
    assert_eq!(repaired.stats.misses, 1);
    assert_eq!(repaired.stats.disk_hits, 0);
    assert_eq!(output.spirv, cold.spirv);
    assert!(output.requires_subgroup_size_32);
    assert_eq!(disk_cache_stats(&temp.0).unwrap().invalid, 0);
}

#[test]
fn geometry_and_tessellation_variants_never_alias() {
    let temp = Temp::new();
    let code = code(false);
    let geom_header = header(4);
    let input = ShaderInput {
        code: &code,
        program_record: &geom_header,
    };
    assert!(Result::from(translate(shadowbox::QuietRun, &input)).is_err());
    let mut keys = HashSet::new();
    for topology in [
        GeometryInput::Points,
        GeometryInput::Lines,
        GeometryInput::LinesAdjacency,
        GeometryInput::Triangles,
        GeometryInput::TrianglesAdjacency,
    ] {
        let context = CacheContext {
            graphics_state: GraphicsState {
                geometry_input: Some(topology),
                tessellation: None,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(keys.insert(identity(&input, context).unwrap()));
        let cold = Result::from(translate_with_options_and_graphics_state(
            shadowbox::QuietRun,
            &input,
            context.options,
            &context.graphics_state,
        ))
        .unwrap();
        let mut first = cache(&temp.0, shadowbox::cache::Keep::nothing());
        let output = Result::from(translate_cached_with_context(
            shadowbox::QuietRun,
            &input,
            context,
            &mut first,
        ))
        .unwrap();
        assert_eq!(first.stats.misses, 1);
        assert_eq!(output.spirv, cold.spirv);
        let mut next = cache(&temp.0, shadowbox::cache::Keep::nothing());
        let hit = Result::from(
            next.translate_with_context(&input, context, || panic!("geometry hit translated")),
        )
        .unwrap();
        assert_eq!(next.stats.disk_hits, 1);
        assert_eq!(hit.spirv, cold.spirv);
    }
    let tess_header = header(3);
    let input = ShaderInput {
        code: &code,
        program_record: &tess_header,
    };
    for domain in [
        TessellationDomain::Isolines,
        TessellationDomain::Triangles,
        TessellationDomain::Quads,
    ] {
        for spacing in [
            TessellationSpacing::Equal,
            TessellationSpacing::FractionalEven,
            TessellationSpacing::FractionalOdd,
        ] {
            for clockwise in [false, true] {
                let context = CacheContext {
                    graphics_state: GraphicsState {
                        geometry_input: None,
                        tessellation: Some(TessellationState {
                            domain,
                            spacing,
                            clockwise,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                };
                assert!(keys.insert(identity(&input, context).unwrap()));
                let cold = Result::from(translate_with_options_and_graphics_state(
                    shadowbox::QuietRun,
                    &input,
                    context.options,
                    &context.graphics_state,
                ))
                .unwrap();
                emit_translation_cache(&temp.0, &input, context, &cold).unwrap();
                let mut next = cache(&temp.0, shadowbox::cache::Keep::nothing());
                let hit = Result::from(
                    next.translate_with_context(&input, context, || panic!("TES hit translated")),
                )
                .unwrap();
                assert_eq!(next.stats.disk_hits, 1);
                assert_eq!(hit.spirv, cold.spirv);
            }
        }
    }
    let mut no_state = cache(&temp.0, shadowbox::cache::Keep::nothing());
    assert!(Result::from(translate_cached_with_context(
        shadowbox::QuietRun,
        &input,
        CacheContext::default(),
        &mut no_state
    ))
    .is_err());
    assert_eq!(no_state.stats.disk_hits, 0);
    assert_eq!(disk_cache_stats(&temp.0).unwrap().entries, 23);
}

#[test]
fn key_accounts_for_header_code_generation_options_and_only_relevant_state() {
    let code = code(false);
    let header = header(0);
    let input = ShaderInput {
        code: &code,
        program_record: &header,
    };
    let default = CacheContext::default();
    let key = identity(&input, default).unwrap();
    let mut keys = HashSet::from([key.clone()]);
    for generation in [TargetGeneration::Sm6, TargetGeneration::Sm7_5Plus] {
        assert!(keys.insert(
            identity(
                &input,
                CacheContext {
                    generation,
                    ..default
                }
            )
            .unwrap()
        ));
    }
    let mut zero = default;
    zero.options.global_delta = shadowbox::GlobalDelta::Zero;
    assert!(keys.insert(identity(&input, zero).unwrap()));
    let mut other = header.clone();
    other[0] ^= 1;
    assert!(keys.insert(
        identity(
            &ShaderInput {
                code: &code,
                program_record: &other
            },
            default
        )
        .unwrap()
    ));
    let mut other_code = code.clone();
    other_code[8] ^= 1;
    assert!(keys.insert(
        identity(
            &ShaderInput {
                code: &other_code,
                program_record: &header
            },
            default
        )
        .unwrap()
    ));
    assert_eq!(
        key,
        identity(
            &ShaderInput {
                code: &code,
                program_record: &header[..80]
            },
            default
        )
        .unwrap()
    );
    let context = CacheContext {
        graphics_state: GraphicsState {
            geometry_input: Some(GeometryInput::Triangles),
            tessellation: Some(TessellationState {
                domain: TessellationDomain::Quads,
                spacing: TessellationSpacing::FractionalOdd,
                clockwise: true,
            }),
            ..Default::default()
        },
        ..default
    };
    assert_eq!(key, identity(&input, context).unwrap());
    let dir = Temp::new();
    let output = Result::from(translate(shadowbox::QuietRun, &input)).unwrap();
    emit_translation_cache(&dir.0, &input, default, &output).unwrap();
    let mut next = cache(&dir.0, shadowbox::cache::Keep::nothing());
    Result::from(next.translate_with_context(&input, context, || {
        panic!("irrelevant state prevented warm hit")
    }))
    .unwrap();
    assert_eq!(next.stats.disk_hits, 1);
}

#[test]
fn concurrent_publish_is_atomic_and_merges_detect_conflicts() {
    let temp = Temp::new();
    let dir = temp.0.join("concurrent");
    let code = code(false);
    let header = header(0);
    let input = ShaderInput {
        code: &code,
        program_record: &header,
    };
    let output = Result::from(translate(shadowbox::QuietRun, &input)).unwrap();
    std::thread::scope(|scope| {
        for _ in 0..16 {
            scope.spawn(|| {
                emit_translation_cache(&dir, &input, CacheContext::default(), &output).unwrap();
            });
        }
    });
    assert_eq!(disk_cache_stats(&dir).unwrap().entries, 1);
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
    let key = identity(&input, CacheContext::default()).unwrap();
    let other = temp.0.join("conflict");
    fs::create_dir(&other).unwrap();
    // Deliberately corrupt metadata but recompute the envelope checksum to model
    // a logically conflicting, structurally valid entry. Never consumed by runtime.
    let mut bytes = fs::read(dir.join(format!("{key}.sbc"))).unwrap();
    bytes[104] ^= 1;
    use sha2::{Digest, Sha256};
    let checksum = Sha256::digest(&bytes[104..]);
    bytes[72..104].copy_from_slice(&checksum);
    fs::write(other.join(format!("{key}.sbc")), bytes).unwrap();
    let destination = temp.0.join("merged");
    assert!(merge_disk_caches(&destination, &[dir.clone(), other.clone()]).is_err());
    assert!(!destination.exists());
    merge_disk_caches(&destination, std::slice::from_ref(&dir)).unwrap();
    assert!(merge_disk_caches(&destination, &[other]).is_err());
    let mut next = cache(&destination, shadowbox::cache::Keep::nothing());
    let hit =
        Result::from(next.translate(&input, Stage::Compute, || panic!("merge lost warm entry")))
            .unwrap();
    assert_eq!(hit.spirv, output.spirv);
    assert!(!hit.requires_subgroup_size_32);
}

// Encoding helpers: Mesa sm50.rs:3418-3428,3106-3110,3178-3184.
fn state_code(first: u64, second: u64) -> Vec<u8> {
    [0, first, second, 0xe300_0000_0007_000f]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect()
}

fn assert_state_variants_match_cold(input: &ShaderInput<'_>, contexts: &[CacheContext<'_>]) {
    let temp = Temp::new();
    let mut warm = cache(&temp.0, shadowbox::cache::Keep::outputs());
    let mut keys = HashSet::new();
    let mut modules = HashSet::new();
    for &context in contexts {
        let cold = Result::from(translate_with_options_and_graphics_state(
            shadowbox::QuietRun,
            input,
            context.options,
            &context.graphics_state,
        ))
        .unwrap();
        assert!(keys.insert(identity(input, context).unwrap()));
        assert!(
            modules.insert(cold.spirv.clone()),
            "state did not exercise a distinct module"
        );
        assert_eq!(
            Result::from(translate_cached_with_context(
                shadowbox::QuietRun,
                input,
                context,
                &mut warm
            ))
            .unwrap()
            .spirv,
            cold.spirv
        );
    }
    assert_eq!(warm.stats.misses, contexts.len());
    for &context in contexts {
        let cold = Result::from(translate_with_options_and_graphics_state(
            shadowbox::QuietRun,
            input,
            context.options,
            &context.graphics_state,
        ))
        .unwrap();
        let mut reopened = cache(&temp.0, shadowbox::cache::Keep::nothing());
        assert_eq!(
            Result::from(translate_cached_with_context(
                shadowbox::QuietRun,
                input,
                context,
                &mut reopened
            ))
            .unwrap()
            .spirv,
            cold.spirv
        );
        assert_eq!(reopened.stats.disk_hits, 1);
    }
}

#[test]
fn bound_texture_shape_cache_hits_match_each_cold_module() {
    // Mesa sm50.rs:2351-2381; Ryujinx InstEmitTexture.cs:1116-1119.
    let txq = 0xdf48_u64 << 48 | 255 << 8 | 7 << 16 | 1 << 22 | 1 << 31 | 4 << 36;
    let code = state_code(txq, 0x50b0_0000_0007_0000);
    let mut h = header(5);
    h[72..76].copy_from_slice(&15u32.to_le_bytes());
    let textures = [TextureDimension::D2, TextureDimension::D3].map(|dimension| BoundTexture {
        binding: 4,
        dimension,
        depth_compare: false,
        multisample: false,
    });
    let contexts: Vec<_> = textures
        .iter()
        .map(|texture| CacheContext {
            graphics_state: GraphicsState {
                texture_bindings: std::slice::from_ref(texture),
                ..Default::default()
            },
            ..Default::default()
        })
        .collect();
    assert_state_variants_match_cold(
        &ShaderInput {
            code: &code,
            program_record: &h,
        },
        &contexts,
    );
}

#[test]
fn linked_patch_layout_and_location_cache_hits_match_each_cold_module() {
    // Mesa sm50.rs:2886-2896; Ryujinx AttributeMap.cs:92,109-121.
    // Byte 0x20 selects different components in the two producer layouts.
    // Mesa sm50.rs:2895 sets bit32, including patch AST. Both values
    // store outputs: Ryujinx InstEmitAttribute.cs:105-141.
    let store = 0xeff0_u64 << 48 | 255 << 8 | 7 << 16 | 0x20 << 20 | 1 << 31 | 1 << 32 | 255 << 39;
    let code = state_code(store, 0x50b0_0000_0007_0000);
    let h = header(2);
    let contexts = [
        (PatchLayout::Ryujinx, 40),
        (PatchLayout::Mesa, 40),
        (PatchLayout::Ryujinx, 41),
    ]
    .map(|(layout, location)| {
        let mut locations = [None; 31];
        locations[0] = Some(location);
        CacheContext {
            graphics_state: GraphicsState {
                patch_io: Some(PatchIo { layout, locations }),
                ..Default::default()
            },
            ..Default::default()
        }
    });
    assert_state_variants_match_cold(
        &ShaderInput {
            code: &code,
            program_record: &h,
        },
        &contexts,
    );
}

#[test]
fn vertex_invocation_info_cache_hits_match_each_topology() {
    // Mesa sm50.rs:3220-3224,2886-2896; Ryujinx InstEmitMove.cs:94-114.
    let info = 0xf0c8_u64 << 48 | 7 << 16 | 0x1d << 20;
    // Mesa sm50.rs:2895 sets bit32. Both values store outputs:
    // Ryujinx InstEmitAttribute.cs:105-141.
    let store = 0xeff0_u64 << 48 | 255 << 8 | 7 << 16 | 0x70 << 20 | 1 << 32 | 255 << 39;
    let code = state_code(info, store);
    let h = header(1);
    let contexts = [GeometryInput::Points, GeometryInput::Triangles].map(|topology| CacheContext {
        graphics_state: GraphicsState {
            geometry_input: Some(topology),
            ..Default::default()
        },
        ..Default::default()
    });
    assert_state_variants_match_cold(
        &ShaderInput {
            code: &code,
            program_record: &h,
        },
        &contexts,
    );
}

#[test]
fn typed_surface_views_survive_memory_hits_and_reject_lossy_disk_writes() {
    // Bound SUST fields: envytools gm107.c:1910. Shape/format identity:
    // Ryujinx Translation/ResourceManager.cs:263-308.
    let surface = (0xeb20_u64 << 48) | (1 << 51) | (9 << 36) | (3 << 33) | (7 << 16) | (2 << 8) | 8;
    let code = [
        0,
        surface | (1 << 20),
        surface | (1 << 52) | (4 << 20),
        0xe300_0000_0007_000f,
    ]
    .into_iter()
    .flat_map(u64::to_le_bytes)
    .collect::<Vec<_>>();
    let header = header(5);
    let input = ShaderInput {
        code: &code,
        program_record: &header,
    };
    let cold = Result::from(translate(shadowbox::QuietRun, &input)).unwrap();
    assert_eq!(cold.surface_descriptors.len(), 2);
    let temp = Temp::new();
    let mut memory = cache(&temp.0, shadowbox::cache::Keep::outputs());
    let first = Result::from(translate_cached_with_context(
        shadowbox::QuietRun,
        &input,
        CacheContext::default(),
        &mut memory,
    ))
    .unwrap();
    let hit =
        Result::from(memory.translate(&input, Stage::Fragment, || panic!("memory hit translated")))
            .unwrap();
    assert_eq!(hit.surface_descriptors, cold.surface_descriptors);
    assert_eq!(hit.spirv, first.spirv);
    assert_eq!(memory.stats.memory_hits, 1);
    assert_eq!(disk_cache_stats(&temp.0).unwrap().entries, 0);
    let error =
        emit_translation_cache(&temp.0, &input, CacheContext::default(), &cold).unwrap_err();
    assert!(error.to_string().contains("surface binding metadata"));
    let mut reopened = cache(&temp.0, shadowbox::cache::Keep::nothing());
    let output = Result::from(translate_cached_with_context(
        shadowbox::QuietRun,
        &input,
        CacheContext::default(),
        &mut reopened,
    ))
    .unwrap();
    assert_eq!(output.surface_descriptors, cold.surface_descriptors);
    assert_eq!(reopened.stats.disk_hits, 0);
}

#[test]
fn geometry_passthrough_link_mask_keys_memory_and_reopened_disk_outputs() {
    // SPH passthrough24: Mesa sph.rs:240-243. Forwarded next-stage inputs:
    // Ryujinx AttributeUsage.cs:91-97 and EmitterContext.cs:360-426.
    let code = code(false);
    let mut header = header(4);
    header[3] |= 1;
    let input = ShaderInput {
        code: &code,
        program_record: &header,
    };
    let temp = Temp::new();
    let mut memory = cache(&temp.0, shadowbox::cache::Keep::outputs());
    let mut keys = Vec::new();
    let mut modules = Vec::new();
    for mask in [1, 4] {
        let context = CacheContext {
            graphics_state: GraphicsState {
                geometry_input: Some(GeometryInput::Points),
                geometry_next_inputs: Some(mask),
                ..Default::default()
            },
            ..Default::default()
        };
        keys.push(identity(&input, context).unwrap());
        let cold = Result::from(translate_with_options_and_graphics_state(
            shadowbox::QuietRun,
            &input,
            context.options,
            &context.graphics_state,
        ))
        .unwrap();
        let first = Result::from(translate_cached_with_context(
            shadowbox::QuietRun,
            &input,
            context,
            &mut memory,
        ))
        .unwrap();
        let hit = Result::from(
            memory.translate_with_context(&input, context, || panic!("memory hit translated")),
        )
        .unwrap();
        let mut reopened = cache(&temp.0, shadowbox::cache::Keep::nothing());
        let disk = Result::from(
            reopened.translate_with_context(&input, context, || panic!("disk hit translated")),
        )
        .unwrap();
        assert_eq!(first.spirv, cold.spirv);
        assert_eq!(hit.spirv, cold.spirv);
        assert_eq!(disk.spirv, cold.spirv);
        modules.push(cold.spirv);
    }
    assert_ne!(keys[0], keys[1]);
    assert_ne!(modules[0], modules[1]);
}

#[test]
fn directed_fmul_cache_separates_exact_float64_from_nearest_fallback() {
    // MOV32I: Mesa sm50.rs:1938-1941. FMUL round39..40:695-700.
    // Float64 host capability: Ryujinx SpirvGenerator.cs:58-60.
    let mov = |dst: u64, value: u32| {
        (0x0100_u64 << 48) | (u64::from(value) << 20) | (0xf << 12) | (7 << 16) | dst
    };
    let nop = 0x50b0_0000_0007_0f00;
    let code = [
        0,
        mov(2, 0x3f80_0001),
        mov(3, 0x3f80_0001),
        (0x5c68_u64 << 48) | (1 << 39) | (7 << 16) | (3 << 20) | (2 << 8) | 1,
        0,
        0xe300_0000_0007_000f,
        nop,
        nop,
    ]
    .into_iter()
    .flat_map(u64::to_le_bytes)
    .collect::<Vec<_>>();
    let header = header(0);
    let input = ShaderInput {
        code: &code,
        program_record: &header,
    };
    let exact = CacheContext {
        options: shadowbox::TranslationOptions {
            float64: shadowbox::HostFloat64::Supported,
            ..Default::default()
        },
        ..Default::default()
    };
    let absent = CacheContext::default();
    assert_ne!(
        identity(&input, exact).unwrap(),
        identity(&input, absent).unwrap()
    );
    let temp = Temp::new();
    let mut memory = cache(&temp.0, shadowbox::cache::Keep::outputs());
    let output = Result::from(translate_cached_with_context(
        shadowbox::QuietRun,
        &input,
        exact,
        &mut memory,
    ))
    .unwrap();
    assert!(output
        .spirv
        .windows(2)
        .any(|words| words == [(2 << 16) | 17, 10]));
    // Ryujinx InstEmitFloatArithmetic.cs:435-480 multiplies in binary32.
    // The fallback must translate and remain a separate cache entry.
    let fallback = Result::from(translate_cached_with_context(
        shadowbox::QuietRun,
        &input,
        absent,
        &mut memory,
    ))
    .unwrap();
    assert!(!fallback
        .spirv
        .windows(2)
        .any(|words| words == [(2 << 16) | 17, 10]));
    assert_ne!(fallback.spirv, output.spirv);
    assert_eq!(
        fallback.spirv,
        Result::from(translate(shadowbox::QuietRun, &input))
            .unwrap()
            .spirv
    );
    let path = temp.0.join("nearest.spv");
    fs::write(
        &path,
        fallback
            .spirv
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let validation = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        validation.status.success(),
        "{}",
        String::from_utf8_lossy(&validation.stderr)
    );
    assert_eq!(
        Result::from(translate_cached_with_context(
            shadowbox::QuietRun,
            &input,
            absent,
            &mut memory
        ))
        .unwrap()
        .spirv,
        fallback.spirv
    );
    let mut disk = cache(&temp.0, shadowbox::cache::Keep::nothing());
    assert_eq!(
        Result::from(translate_cached_with_context(
            shadowbox::QuietRun,
            &input,
            absent,
            &mut disk
        ))
        .unwrap()
        .spirv,
        fallback.spirv
    );
    assert_eq!(disk.stats.disk_hits, 1);
    assert_eq!(memory.stats.memory_hits, 1);
}

// A reopened cache must keep the host feature requirement of token modules.
// Token fields: Mesa sm50.rs:3038-3051; VOTE: :3233-3247. Feature/mode:
// vulkan_core.h:13103-13111; Mesa spirv.h:210.
#[test]
fn token_region_feature_requirement_survives_disk_cache() {
    let temp = Temp::new();
    let pt = 7u64 << 16;
    let vote = 0x50d8u64 << 48 | 7 << 39 | 7 << 45 | pt;
    let code = [
        0,
        0xe290u64 << 48 | 32 << 20 | pt | 15,
        0xf0f8u64 << 48 | 15,
        vote,
        0,
        0xf0f8u64 << 48 | pt | 15,
        vote,
        0xe300u64 << 48 | pt | 15,
    ]
    .into_iter()
    .flat_map(u64::to_le_bytes)
    .collect::<Vec<_>>();
    let header = header(0);
    let input = ShaderInput {
        code: &code,
        program_record: &header,
    };
    let mut writer = cache(&temp.0, shadowbox::cache::Keep::outputs());
    let cold = Result::from(translate_cached_with_context(
        shadowbox::QuietRun,
        &input,
        CacheContext::default(),
        &mut writer,
    ))
    .unwrap();
    assert!(cold.requires_maximal_reconvergence);
    assert!(cold.requires_subgroup_size_32);
    let mut reader = cache(&temp.0, shadowbox::cache::Keep::nothing());
    let hit =
        Result::from(reader.translate(&input, Stage::Compute, || panic!("disk hit translated")))
            .unwrap();
    assert!(hit.requires_maximal_reconvergence);
    assert!(hit.requires_subgroup_size_32);
    assert_eq!(hit.spirv, cold.spirv);
    assert_eq!(reader.stats.disk_hits, 1);
}

#[test]
fn geometry_stream_routing_changes_modules_and_cache_identity() {
    use shadowbox::interface::{GeometryStreamLocations, GeometryStreams};
    let temp = Temp::new();
    // AST: Mesa sm50.rs:2886-2896. OUT.I:3263-3284. Bundle:3418-3428.
    let ast = |address: u64| {
        (0xeff0u64 << 48) | 255 | (255 << 8) | (7 << 16) | (address << 20) | (1 << 32) | (255 << 39)
    };
    let out =
        |stream: u64| (0xf6e0u64 << 48) | 255 | (255 << 8) | (7 << 16) | (stream << 20) | (3 << 39);
    let code = [
        0,
        ast(0x80),
        ast(0x60),
        out(0),
        0,
        out(1),
        0xe300_0000_0007_000f,
        0x50b0_0000_0007_0f00,
    ]
    .into_iter()
    .flat_map(u64::to_le_bytes)
    .collect::<Vec<_>>();
    let mut h = header(4);
    h[16] = 2;
    let input = ShaderInput {
        code: &code,
        program_record: &h,
    };
    let base = GraphicsState {
        geometry_input: Some(GeometryInput::Points),
        geometry_streams: Some(GeometryStreams {
            builtin_stream: 0,
            locations: [
                Some(GeometryStreamLocations {
                    generic_base: 0,
                    position: None,
                    primitive_id: None,
                }),
                Some(GeometryStreamLocations {
                    generic_base: 4,
                    position: Some(5),
                    primitive_id: Some(6),
                }),
                None,
                None,
            ],
        }),
        ..Default::default()
    };
    let mut keys = HashSet::new();
    let mut modules = HashSet::new();
    let mut memory = cache(&temp.0, shadowbox::cache::Keep::outputs());
    for variant in 0..5 {
        let mut state = base;
        let layout = state.geometry_streams.as_mut().unwrap();
        match variant {
            0 => {}
            1 => layout.locations[1].as_mut().unwrap().generic_base = 7,
            2 => layout.locations[1].as_mut().unwrap().position = Some(7),
            3 => layout.locations[1].as_mut().unwrap().primitive_id = Some(7),
            4 => {
                layout.builtin_stream = 1;
                layout.locations[0].as_mut().unwrap().position = Some(1);
                layout.locations[0].as_mut().unwrap().primitive_id = Some(2);
                layout.locations[1].as_mut().unwrap().position = None;
                layout.locations[1].as_mut().unwrap().primitive_id = None;
            }
            _ => unreachable!(),
        }
        let context = CacheContext {
            graphics_state: state,
            ..Default::default()
        };
        assert!(keys.insert(identity(&input, context).unwrap()));
        let cold = Result::from(translate_with_options_and_graphics_state(
            shadowbox::QuietRun,
            &input,
            context.options,
            &state,
        ))
        .unwrap();
        assert!(modules.insert(cold.spirv.clone()));
        let first = Result::from(translate_cached_with_context(
            shadowbox::QuietRun,
            &input,
            context,
            &mut memory,
        ))
        .unwrap();
        let hit = Result::from(
            memory.translate_with_context(&input, context, || panic!("memory hit translated")),
        )
        .unwrap();
        let mut disk = cache(&temp.0, shadowbox::cache::Keep::nothing());
        let hit_disk = Result::from(
            disk.translate_with_context(&input, context, || panic!("disk hit translated")),
        )
        .unwrap();
        assert_eq!(first.spirv, cold.spirv);
        assert_eq!(hit.spirv, cold.spirv);
        assert_eq!(hit_disk.spirv, cold.spirv);
    }
    // A removed layout must not reuse a previously cached successful module.
    let mut state = base;
    state.geometry_streams.as_mut().unwrap().locations[1] = None;
    let context = CacheContext {
        graphics_state: state,
        ..Default::default()
    };
    assert!(!keys.contains(&identity(&input, context).unwrap()));
    assert!(Result::from(translate_cached_with_context(
        shadowbox::QuietRun,
        &input,
        context,
        &mut memory
    ))
    .is_err());
}

// Project v8 envelope offsets: docs/reference-sources.md.
fn checksum_envelope(bytes: &mut [u8]) {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(&bytes[104..]);
    bytes[72..104].copy_from_slice(&digest);
}

#[test]
fn foreign_build_identity_is_never_served_even_under_the_requested_key() {
    use sha2::{Digest, Sha256};
    use shadowbox::{translator_identity, verify_disk_cache, CacheEntryStatus};
    let temp = Temp::new();
    let code = code(false);
    let header = header(0);
    let input = ShaderInput {
        code: &code,
        program_record: &header,
    };
    let context = CacheContext::default();
    let output = Result::from(translate(shadowbox::QuietRun, &input)).unwrap();
    let key = emit_translation_cache(&temp.0, &input, context, &output).unwrap();
    let path = temp.0.join(format!("{key}.sbc"));
    let mut bytes = fs::read(&path).unwrap();
    // Model a writer built from different decoder/emitter sources, with a valid
    // checksum. The requested filename deliberately tests the envelope guard.
    let mut foreign = Sha256::new();
    foreign.update(b"shadowbox-translator-identity-v1");
    foreign.update(format!("{:x}", Sha256::digest(b"different translator sources")).as_bytes());
    foreign.update(shadowbox::CACHE_FORMAT_VERSION.to_le_bytes());
    foreign.update([0, 0, 0]);
    bytes[105..169].copy_from_slice(format!("{:x}", foreign.finalize()).as_bytes());
    assert_ne!(
        &bytes[105..169],
        translator_identity(context.generation, context.options).as_bytes()
    );
    checksum_envelope(&mut bytes);
    fs::write(&path, bytes).unwrap();
    assert_eq!(
        verify_disk_cache(&temp.0).unwrap(),
        vec![(format!("{key}.sbc"), CacheEntryStatus::Mismatched)]
    );
    let merged = temp.0.join("merged");
    assert!(merge_disk_caches(&merged, std::slice::from_ref(&temp.0)).is_err());
    assert!(!merged.exists());
    let mut runtime = cache(&temp.0, shadowbox::cache::Keep::outputs());
    let repaired = Result::from(
        runtime.translate_with_context(&input, context, || Translation::Module(output.clone())),
    )
    .unwrap();
    assert_eq!(runtime.stats.disk_hits, 0);
    assert_eq!(runtime.stats.misses, 1);
    assert_eq!(repaired.spirv, output.spirv);
    Result::from(
        runtime.translate_with_context(&input, context, || panic!("repaired memory entry missed")),
    )
    .unwrap();
    let mut disk = cache(&temp.0, shadowbox::cache::Keep::nothing());
    Result::from(
        disk.translate_with_context(&input, context, || panic!("repaired disk entry missed")),
    )
    .unwrap();
    assert_eq!(disk.stats.disk_hits, 1);
}

#[test]
fn output_option_identities_are_distinct_and_verify_with_their_recorded_options() {
    use shadowbox::{translator_identity, verify_disk_cache, CacheEntryStatus, TranslationOptions};
    let temp = Temp::new();
    let code = code(false);
    let header = header(0);
    let input = ShaderInput {
        code: &code,
        program_record: &header,
    };
    let output = Result::from(translate(shadowbox::QuietRun, &input)).unwrap();
    let mut identities = HashSet::new();
    for generation in [
        TargetGeneration::Sm5,
        TargetGeneration::Sm6,
        TargetGeneration::Sm7_5Plus,
    ] {
        for global_delta in [shadowbox::GlobalDelta::Pushed, shadowbox::GlobalDelta::Zero] {
            for float64 in [
                shadowbox::HostFloat64::Unsupported,
                shadowbox::HostFloat64::Supported,
            ] {
                let options = TranslationOptions {
                    global_delta,
                    float64,
                };
                assert!(identities.insert(translator_identity(generation, options)));
                let context = CacheContext {
                    generation,
                    options,
                    ..Default::default()
                };
                emit_translation_cache(&temp.0, &input, context, &output).unwrap();
                Result::from(
                    cache(&temp.0, shadowbox::cache::Keep::nothing()).translate_with_context(
                        &input,
                        context,
                        || panic!("option variant missed"),
                    ),
                )
                .unwrap();
            }
        }
    }
    assert_eq!(disk_cache_stats(&temp.0).unwrap().entries, 12);
    assert!(verify_disk_cache(&temp.0)
        .unwrap()
        .iter()
        .all(|(_, s)| *s == CacheEntryStatus::Valid));
    // A valid entry for other output options cannot be served under the default key.
    let default = CacheContext::default();
    let other = CacheContext {
        options: TranslationOptions {
            global_delta: shadowbox::GlobalDelta::Zero,
            ..Default::default()
        },
        ..default
    };
    let key = identity(&input, default).unwrap();
    let other_key = identity(&input, other).unwrap();
    let mut bytes = fs::read(temp.0.join(format!("{other_key}.sbc"))).unwrap();
    bytes[8..72].copy_from_slice(key.as_bytes());
    fs::write(temp.0.join(format!("{key}.sbc")), bytes).unwrap();
    let mut runtime = cache(&temp.0, shadowbox::cache::Keep::nothing());
    Result::from(runtime.translate_with_context(&input, default, || Translation::Module(output)))
        .unwrap();
    assert_eq!(runtime.stats.misses, 1);
    assert_eq!(runtime.stats.disk_hits, 0);
}

#[test]
fn legacy_format_is_stale_and_damage_is_corrupt() {
    use shadowbox::{verify_disk_cache, CacheEntryStatus};
    let temp = Temp::new();
    let code = code(false);
    let header = header(0);
    let input = ShaderInput {
        code: &code,
        program_record: &header,
    };
    let output = Result::from(translate(shadowbox::QuietRun, &input)).unwrap();
    let key = emit_translation_cache(&temp.0, &input, CacheContext::default(), &output).unwrap();
    let path = temp.0.join(format!("{key}.sbc"));
    let current = fs::read(&path).unwrap();
    let mut legacy = current.clone();
    legacy[..8].copy_from_slice(b"SBOXC007");
    legacy.drain(105..172);
    checksum_envelope(&mut legacy);
    fs::write(&path, &legacy).unwrap();
    assert_eq!(
        verify_disk_cache(&temp.0).unwrap()[0].1,
        CacheEntryStatus::Stale
    );
    assert!(merge_disk_caches(&temp.0.join("merge"), std::slice::from_ref(&temp.0)).is_err());
    let mut runtime = cache(&temp.0, shadowbox::cache::Keep::nothing());
    Result::from(
        runtime.translate_with_context(&input, CacheContext::default(), || {
            Translation::Module(output)
        }),
    )
    .unwrap();
    assert_eq!(runtime.stats.misses, 1);
    assert_eq!(runtime.stats.disk_hits, 0);
    // Malformed identity, out-of-range option and damaged checksum are corruption,
    // even if another field would otherwise identify a different translator.
    for offset in [105, 169, 170, 171, 172] {
        let mut bytes = current.clone();
        bytes[offset] = 255;
        if offset != 172 {
            checksum_envelope(&mut bytes);
        }
        fs::write(&path, bytes).unwrap();
        assert_eq!(
            verify_disk_cache(&temp.0).unwrap()[0].1,
            CacheEntryStatus::Corrupt
        );
    }
    for length in 0..current.len() {
        fs::write(&path, &current[..length]).unwrap();
        assert_eq!(
            verify_disk_cache(&temp.0).unwrap()[0].1,
            CacheEntryStatus::Corrupt
        );
    }
}

#[test]
fn demote_requirement_survives_memory_and_disk_cache() {
    // Mesa sm50.rs:3172-3174 and from_nir.rs:3035-3050: KIL=demote.
    let raw = [
        0u64,
        0xe330_0000_0007_000f,
        0xe300_0000_0007_000f,
        0x50b0_0000_0007_0f00,
    ]
    .into_iter()
    .flat_map(u64::to_le_bytes)
    .collect::<Vec<_>>();
    let input = ShaderInput {
        code: &raw,
        program_record: &[],
    };
    let temp = Temp::new();
    let mut first = cache(&temp.0, shadowbox::cache::Keep::outputs());
    let cold = Result::from(translate_cached_with_context(
        shadowbox::QuietRun,
        &input,
        CacheContext::default(),
        &mut first,
    ))
    .unwrap();
    assert!(cold.requires_demote_to_helper_invocation);
    let hot =
        Result::from(first.translate(&input, Stage::Fragment, || panic!("memory hit translated")))
            .unwrap();
    assert!(hot.requires_demote_to_helper_invocation);
    let mut second = cache(&temp.0, shadowbox::cache::Keep::nothing());
    let disk =
        Result::from(second.translate(&input, Stage::Fragment, || panic!("disk hit translated")))
            .unwrap();
    assert_eq!(second.stats.disk_hits, 1);
    assert!(disk.requires_demote_to_helper_invocation);
    assert_eq!(cold.spirv, disk.spirv);
}

#[test]
fn v8_requirement_bits_round_trip_and_unknown_bits_are_corrupt() {
    use shadowbox::{
        cache::{decode_record, encode_record},
        CacheEntryStatus,
    };
    let raw = code(false);
    let hdr = header(0);
    let input = ShaderInput {
        code: &raw,
        program_record: &hdr,
    };
    let mut output = Result::from(translate(shadowbox::QuietRun, &input)).unwrap();
    let context = CacheContext::default();
    let key = identity(&input, context).unwrap();
    for flags in 0..=7 {
        output.requires_subgroup_size_32 = flags & 1 != 0;
        output.requires_maximal_reconvergence = flags & 2 != 0;
        output.requires_demote_to_helper_invocation = flags & 4 != 0;
        let bytes = encode_record(&key, context, &output).unwrap();
        assert_eq!(&bytes[..8], b"SBOXC008");
        assert_eq!(bytes[104], flags);
        let restored = decode_record(&bytes, &key).unwrap();
        assert_eq!(restored.spirv, output.spirv);
        assert_eq!(
            restored.requires_subgroup_size_32,
            output.requires_subgroup_size_32
        );
        assert_eq!(
            restored.requires_maximal_reconvergence,
            output.requires_maximal_reconvergence
        );
        assert_eq!(
            restored.requires_demote_to_helper_invocation,
            output.requires_demote_to_helper_invocation
        );
    }
    let valid = encode_record(&key, context, &output).unwrap();
    for flags in 8..=255 {
        let mut bytes = valid.clone();
        bytes[104] = flags;
        checksum_envelope(&mut bytes);
        assert!(matches!(
            decode_record(&bytes, &key),
            Err(CacheEntryStatus::Corrupt)
        ));
    }
}
