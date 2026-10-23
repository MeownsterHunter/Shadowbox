//! Required shader extensions for the exact Vulkan transports.
//! Query both extension advertisement and feature bits before enabling them.
//! /usr/include/vulkan/vulkan_core.h:7747-7751,11877-11883,13103-13111.
use ash::vk;
use shadowbox::spirv::Op;
use std::ffi::CStr;

#[derive(Clone, Copy, Default)]
pub struct ShaderFeatures {
    pub maximal_reconvergence: bool,
    pub quad_control: bool,
    pub demote_to_helper_invocation: bool,
    demote_extension: bool,
}
impl ShaderFeatures {
    pub unsafe fn query(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Self {
        let extensions =
            unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap();
        let has = |name| {
            extensions
                .iter()
                .any(|e| e.extension_name_as_c_str().unwrap() == name)
        };
        let mut maximal = vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default();
        let mut quad = vk::PhysicalDeviceShaderQuadControlFeaturesKHR::default();
        // Core since Vulkan 1.3. Query the feature even without extension
        // advertisement; enabling a feature still requires its actual bit.
        // Vulkan-Docs v1.4.351 features.adoc:3910-3913;
        // vulkan_core.h:7747-7751,19635-19636.
        let demote_extension = has(ash::ext::shader_demote_to_helper_invocation::NAME);
        let demote_available = demote_extension
            || unsafe { instance.get_physical_device_properties(physical) }.api_version
                >= vk::API_VERSION_1_3;
        let mut demote = vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures::default();
        let mut query = vk::PhysicalDeviceFeatures2::default();
        if demote_available {
            query = query.push_next(&mut demote);
        }
        if has(ash::khr::shader_maximal_reconvergence::NAME) {
            query = query.push_next(&mut maximal);
        }
        if has(ash::khr::shader_quad_control::NAME) {
            query = query.push_next(&mut quad);
        }
        unsafe { instance.get_physical_device_features2(physical, &mut query) };
        Self {
            maximal_reconvergence: maximal.shader_maximal_reconvergence != 0,
            quad_control: quad.shader_quad_control != 0,
            demote_to_helper_invocation: demote.shader_demote_to_helper_invocation != 0,
            demote_extension,
        }
    }
    pub fn extension_names(self) -> Vec<*const std::ffi::c_char> {
        let mut names = Vec::new();
        if self.maximal_reconvergence {
            names.push(ash::khr::shader_maximal_reconvergence::NAME.as_ptr());
        }
        if self.quad_control {
            names.push(ash::khr::shader_quad_control::NAME.as_ptr());
        }
        if self.demote_to_helper_invocation && self.demote_extension {
            names.push(ash::ext::shader_demote_to_helper_invocation::NAME.as_ptr());
        }
        names
    }
    pub fn describe(self) -> String {
        format!("VK_KHR_shader_maximal_reconvergence/shaderMaximalReconvergence={} VK_KHR_shader_quad_control/shaderQuadControl={} shaderDemoteToHelperInvocation={}",
            self.maximal_reconvergence, self.quad_control, self.demote_to_helper_invocation)
    }
    pub fn require_module(self, words: &[u32]) {
        let mut at = 5;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            assert!(count > 0 && at + count <= words.len(), "malformed SPIR-V");
            // Mesa src/compiler/spirv/spirv.h:1746,1751: OpExtension=10,
            // OpExecutionMode=16. Check the actual module, including GLSL
            // references, rather than relying on a caller-provided flag.
            // Check capability too, so SPIR-V 1.6 modules using the core
            // instruction cannot bypass the extension-string check.
            // Mesa spirv.h:1272-1273,2310-2312.
            if (Op::of(words[at]) == Op::Capability && words[at + 1] == 5379)
                || matches!(
                    Op::of(words[at]),
                    Op::DemoteToHelperInvocation | Op::IsHelperInvocationEXT
                )
            {
                assert!(self.demote_to_helper_invocation,
                    "device lacks required shaderDemoteToHelperInvocation (VK_EXT_shader_demote_to_helper_invocation or Vulkan 1.3)");
            }
            if Op::of(words[at]) == Op::Extension {
                let bytes = words[at + 1..at + count]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>();
                let name = CStr::from_bytes_until_nul(&bytes).unwrap().to_bytes();
                if name == b"SPV_KHR_maximal_reconvergence" {
                    assert!(self.maximal_reconvergence,
                        "device lacks required VK_KHR_shader_maximal_reconvergence or shaderMaximalReconvergence");
                }
                if name == b"SPV_KHR_quad_control" {
                    assert!(
                        self.quad_control,
                        "device lacks required VK_KHR_shader_quad_control or shaderQuadControl"
                    );
                    assert!(
                        self.maximal_reconvergence,
                        "SPV_KHR_quad_control requires VK_KHR_shader_maximal_reconvergence"
                    );
                }
            }
            at += count;
        }
    }
}

#[test]
#[should_panic(expected = "device lacks required shaderDemoteToHelperInvocation")]
fn missing_demote_feature_rejects_core_spirv_capability() {
    // SPIR-V 1.6 can omit OpExtension, but still requires the Vulkan feature.
    ShaderFeatures::default().require_module(&[
        0x0723_0203,
        0x0001_0600,
        0,
        1,
        0,
        2 << 16 | 17,
        5379,
    ]);
}
