// Resource shape follows Ryujinx CodeGen/Spirv/Declarations.cs:157-221.
// Separate image/sampler composition and sampling follow Instructions.cs:
// 1400-1445,1908-1948. No upstream implementation is copied.
// SPIR-V fields: /usr/include/spirv/unified1/spirv.h:270-279,1797-1799,
// 1852-1856; operand ordering: spirv.core.grammar.json:354-390,984-1051.
// Extents, texels and sampler policy are project-owned test inputs.
use super::{graphics::Scalar, Analysis, Type};
use ash::vk;
use shadowbox::spirv::Op;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::value_diff) struct ImageType {
    pub dim: u32,
    pub arrayed: bool,
    pub depth: bool,
}

impl ImageType {
    pub fn extent(self) -> vk::Extent3D {
        vk::Extent3D {
            width: 4,
            height: if self.dim == 0 { 1 } else { 4 },
            depth: if self.dim == 2 { 4 } else { 1 },
        }
    }
    pub fn layers(self) -> u32 {
        (if self.dim == 3 { 6 } else { 1 }) * if self.arrayed { 2 } else { 1 }
    }
    pub fn coords(self) -> usize {
        (match self.dim {
            0 => 1,
            1 => 2,
            _ => 3,
        }) + usize::from(self.arrayed)
    }
    // Format and view enums: vulkan_core.h:1968,1985,2327-2333.
    pub fn format(self) -> vk::Format {
        if self.depth {
            vk::Format::D32_SFLOAT
        } else {
            vk::Format::R32G32B32A32_SFLOAT
        }
    }
    pub fn view(self) -> vk::ImageViewType {
        match (self.dim, self.arrayed) {
            (0, false) => vk::ImageViewType::TYPE_1D,
            (0, true) => vk::ImageViewType::TYPE_1D_ARRAY,
            (1, false) => vk::ImageViewType::TYPE_2D,
            (1, true) => vk::ImageViewType::TYPE_2D_ARRAY,
            (2, false) => vk::ImageViewType::TYPE_3D,
            (3, false) => vk::ImageViewType::CUBE,
            (3, true) => vk::ImageViewType::CUBE_ARRAY,
            _ => unreachable!(),
        }
    }
    // Image aspects: vulkan_core.h:2966-2984.
    pub fn aspect(self) -> vk::ImageAspectFlags {
        if self.depth {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        }
    }
    pub fn texels(self, set: u32, binding: u32) -> Vec<u8> {
        let e = self.extent();
        let count = e.width * e.height * e.depth * self.layers();
        // Exact finite binary fractions, differing by texel, component and
        // descriptor. Depth stays in (0,1). Pattern is independent of seed.
        // A single level avoids derivative-dependent mip choices.
        (0..count)
            .flat_map(|texel| {
                (0..if self.depth { 1 } else { 4 }).flat_map(move |component| {
                    let value = 1 + (texel * 17 + component * 29 + set * 13 + binding * 7) % 251;
                    (value as f32 / 256.0).to_le_bytes()
                })
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::value_diff) enum ResourceKind {
    Image { ty: ImageType, combined: bool },
    Sampler { compare: Option<bool> },
}
impl ResourceKind {
    // Descriptor enums: vulkan_core.h:2410-2412.
    pub fn descriptor(self) -> vk::DescriptorType {
        match self {
            Self::Image { combined: true, .. } => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            Self::Image {
                combined: false, ..
            } => vk::DescriptorType::SAMPLED_IMAGE,
            Self::Sampler { .. } => vk::DescriptorType::SAMPLER,
        }
    }
}
pub(in crate::value_diff) struct Resource {
    pub set: u32,
    pub binding: u32,
    pub kind: ResourceKind,
}
#[derive(Clone, Copy)]
pub(super) struct Handle {
    ty: u32,
    image: Option<u32>,
    sampler: Option<u32>,
}

pub(super) fn declare(a: &mut Analysis, op: Op, args: &[u32]) -> Result<(), String> {
    let ty = match op {
        Op::TypeImage => {
            if args.len() != 8
                || a.scalars.get(&args[1]) != Some(&Scalar::Float)
                || args[2] > 3
                || args[3] > 1
                || args[4] > 1
                || args[5] != 0
                || args[6] != 1
                || args[7] != 0
                || args[2] == 2 && (args[3] != 0 || args[4] != 0)
            {
                return Err("unsupported sampled image type, multisample or format".into());
            }
            Type::Image(ImageType {
                dim: args[2],
                arrayed: args[4] == 1,
                depth: args[3] == 1,
            })
        }
        Op::TypeSampler if args.len() == 1 => Type::Sampler,
        Op::TypeSampledImage if args.len() == 2 && matches!(a.ty(args[1])?, Type::Image(_)) => {
            Type::SampledImage(args[1])
        }
        _ => return Err("unsupported opaque type".into()),
    };
    a.types.insert(args[0], ty);
    Ok(())
}

pub(super) fn variable(
    a: &mut Analysis,
    id: u32,
    ty: u32,
    init: Option<u32>,
) -> Result<(), String> {
    if init.is_some() {
        return Err("opaque initializer unsupported".into());
    }
    let (kind, handle) = match *a.ty(ty)? {
        Type::Image(image) => (
            ResourceKind::Image {
                ty: image,
                combined: false,
            },
            Handle {
                ty,
                image: Some(id),
                sampler: None,
            },
        ),
        Type::SampledImage(image) => {
            let Type::Image(image) = *a.ty(image)? else {
                return Err("invalid sampled image".into());
            };
            (
                ResourceKind::Image {
                    ty: image,
                    combined: true,
                },
                Handle {
                    ty,
                    image: Some(id),
                    sampler: Some(id),
                },
            )
        }
        Type::Sampler => (
            ResourceKind::Sampler { compare: None },
            Handle {
                ty,
                image: None,
                sampler: Some(id),
            },
        ),
        _ => return Err("opaque descriptor arrays unsupported".into()),
    };
    let &(set, binding) = a
        .bindings
        .get(&id)
        .ok_or("image/sampler lacks descriptor coordinates")?;
    let key = (
        set.ok_or("image/sampler lacks set")?,
        binding.ok_or("image/sampler lacks binding")?,
    );
    if key.0 > 3 || key.1 > 255 || a.resources.len() == 16 {
        return Err("image/sampler exceeds harness limits".into());
    }
    if a.roots.contains_key(&key) || a.resources.insert(key, (id, kind)).is_some() {
        return Err("duplicate descriptor binding".into());
    }
    a.handles.insert(id, handle);
    Ok(())
}

pub(super) fn load(a: &mut Analysis, args: &[u32]) -> Result<(), String> {
    // Both OpLoad and OpCopyObject have result type, result ID, operand ID.
    if args.len() != 3 {
        return Err("opaque load/copy operands unsupported".into());
    }
    let handle = a.handles[&args[2]];
    if handle.ty != args[0] {
        return Err("opaque load/copy type mismatch".into());
    }
    a.handles.insert(args[1], handle);
    Ok(())
}

pub(super) fn instruction(a: &mut Analysis, op: Op, args: &[u32]) -> Result<(), String> {
    if op == Op::SampledImage {
        if args.len() != 4 {
            return Err("sampled image operand count".into());
        }
        let image = *a.handles.get(&args[2]).ok_or("missing image handle")?;
        let sampler = *a.handles.get(&args[3]).ok_or("missing sampler handle")?;
        if !matches!(a.ty(args[0])?, Type::SampledImage(ty) if *ty == image.ty)
            || !matches!(a.ty(sampler.ty)?, Type::Sampler)
        {
            return Err("image/sampler type mismatch".into());
        }
        a.handles.insert(
            args[1],
            Handle {
                ty: args[0],
                image: image.image,
                sampler: sampler.sampler,
            },
        );
        return Ok(());
    }
    let handle = *a.handles.get(&args[2]).ok_or("missing sampled image")?;
    let Type::SampledImage(image_ty) = *a.ty(handle.ty)? else {
        return Err("sample requires sampled image".into());
    };
    let Type::Image(image) = *a.ty(image_ty)? else {
        return Err("sample image type missing".into());
    };
    let dref = matches!(
        op,
        Op::ImageSampleDrefImplicitLod | Op::ImageSampleDrefExplicitLod
    );
    if dref != image.depth || handle.image.is_none() || handle.sampler.is_none() {
        return Err("sample depth or handle mismatch".into());
    }
    let base = if dref { 5 } else { 4 };
    if matches!(
        op,
        Op::ImageSampleExplicitLod | Op::ImageSampleDrefExplicitLod
    ) {
        // Only explicit constant LOD zero. Other operand masks retain their
        // original instructions but cannot yet receive a definedness proof.
        // ImageOperands Lod bit 2: spirv.h:397-430; grammar:1004-1051.
        if args.len() != base + 2 || args[base] != 2 || a.constants.get(&args[base + 1]) != Some(&0)
        {
            return Err("only explicit zero LOD is supported".into());
        }
    } else if args.len() != base {
        return Err("implicit sample image operands unsupported".into());
    }
    let coords = a
        .literals
        .get(&args[3])
        .ok_or("sample coordinates must be proved constant")?;
    if coords.len() != image.coords()
        || coords.iter().any(|&bits| !f32::from_bits(bits).is_finite())
        || image.dim == 3 && coords[..3].iter().all(|&bits| f32::from_bits(bits) == 0.0)
    {
        return Err("sample coordinates are nonfinite, zero cube direction or wrong extent".into());
    }
    if dref {
        let bits = a
            .constants
            .get(&args[4])
            .ok_or("depth reference must be constant")?;
        if !f32::from_bits(*bits).is_finite() {
            return Err("nonfinite depth reference".into());
        }
    }
    // A sampler cannot switch comparison policy across uses. An unused
    // sampler receives ordinary nearest filtering when allocated.
    let sampler = handle.sampler.unwrap();
    for (root, kind) in a.resources.values_mut() {
        if *root == sampler {
            if let ResourceKind::Sampler { compare } = kind {
                if compare.is_some_and(|previous| previous != dref) {
                    return Err("sampler comparison policy conflict".into());
                }
                *compare = Some(dref);
            }
        }
    }
    let len = a.size(args[0])?;
    if len != if dref { 4 } else { 16 } {
        return Err("sample result extent mismatch".into());
    }
    a.result(args[0], args[1], vec![true; len])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_image_fields_and_optional_access_qualifier_fail_closed() {
        let baseline = [10, 1, 1, 0, 0, 0, 1, 0];
        let mut a = Analysis::default();
        a.scalars.insert(1, Scalar::Float);
        declare(&mut a, Op::TypeImage, &baseline).unwrap();
        for (field, value) in [(1, 2), (2, 4), (3, 2), (4, 2), (5, 1), (6, 2), (7, 1)] {
            let mut args = baseline;
            args[field] = value;
            assert!(
                declare(&mut a, Op::TypeImage, &args).is_err(),
                "field {field}"
            );
        }
        let mut args = baseline.to_vec();
        args.push(0);
        assert!(declare(&mut a, Op::TypeImage, &args).is_err());
    }

    #[test]
    fn texel_pattern_distinguishes_channels_layers_and_descriptors() {
        let ty = ImageType {
            dim: 1,
            arrayed: true,
            depth: false,
        };
        let bytes = ty.texels(1, 14);
        assert_eq!(bytes.len(), 4 * 4 * 2 * 16);
        assert_ne!(&bytes[..4], &bytes[4..8]);
        assert_ne!(&bytes[..16], &bytes[4 * 4 * 16..4 * 4 * 16 + 16]);
        assert_ne!(bytes, ty.texels(1, 16));
    }
}
