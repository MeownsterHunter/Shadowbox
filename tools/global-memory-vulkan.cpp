// Project-owned Vulkan execution check for flat global memory.
// Vulkan API definitions: /usr/include/vulkan/vulkan_core.h:6695-6707,
// 3067,5567,6021,6526,3567. Shader/encoding sources are in the Rust test.
#include <vulkan/vulkan.h>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(VkResult result) {
    if (result != VK_SUCCESS) throw std::runtime_error("Vulkan result " + std::to_string(result));
}
static std::vector<char> read(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error(std::string("Cannot read ") + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
struct Buffer {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    void* mapped{};
};
int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) return 2;
    try {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Shadowbox global-memory check";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance_info.pApplicationInfo = &app;
        VkInstance instance{};
        check(vkCreateInstance(&instance_info, nullptr, &instance));
        uint32_t device_count{};
        check(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));
        std::vector<VkPhysicalDevice> physical_devices(device_count);
        check(vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data()));
        if (device_count == 0) return 77;
        VkPhysicalDevice physical{};
        uint32_t queue_index{};
        for (auto candidate : physical_devices) {
            VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
            f11.pNext = &f12;
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext = &f11;
            vkGetPhysicalDeviceFeatures2(candidate, &features);
            if (!f12.bufferDeviceAddress || !features.features.shaderInt64 ||
                !features.features.shaderInt16 || !f12.shaderInt8 ||
                !f11.storageBuffer16BitAccess || !f12.storageBuffer8BitAccess) continue;
            uint32_t count{};
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
            std::vector<VkQueueFamilyProperties> queues(count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, queues.data());
            for (uint32_t i = 0; i < count; ++i) {
                if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physical = candidate;
                    queue_index = i;
                    break;
                }
            }
            if (physical) break;
        }
        if (!physical) return 77;
        float priority = 1;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = queue_index;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.bufferDeviceAddress = VK_TRUE;
        f12.storageBuffer8BitAccess = VK_TRUE;
        f12.shaderInt8 = VK_TRUE;
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        f11.pNext = &f12;
        f11.storageBuffer16BitAccess = VK_TRUE;
        VkPhysicalDeviceFeatures features{};
        features.shaderInt64 = VK_TRUE;
        features.shaderInt16 = VK_TRUE;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.pNext = &f11;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.pEnabledFeatures = &features;
        VkDevice device{};
        check(vkCreateDevice(physical, &device_info, nullptr, &device));
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
        auto make_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, bool address) {
            Buffer buffer{};
            VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size = size;
            info.usage = usage;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            check(vkCreateBuffer(device, &info, nullptr, &buffer.buffer));
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
            uint32_t type = UINT32_MAX;
            for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
                constexpr auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if ((requirements.memoryTypeBits & (1u << i)) &&
                    (memory_properties.memoryTypes[i].propertyFlags & flags) == flags) { type = i; break; }
            }
            if (type == UINT32_MAX) throw std::runtime_error("No coherent host-visible memory");
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocate.pNext = address ? &flags : nullptr;
            allocate.allocationSize = requirements.size;
            allocate.memoryTypeIndex = type;
            check(vkAllocateMemory(device, &allocate, nullptr, &buffer.memory));
            check(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0));
            check(vkMapMemory(device, buffer.memory, 0, size, 0, &buffer.mapped));
            return buffer;
        };
        auto memory = make_buffer(128, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
        auto cbuf = make_buffer(65536, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, false);
        auto initial = read(argv[2]);
        auto expected = read(argv[3]);
        if (initial.size() != 128 || expected.size() != 128) throw std::runtime_error("Expected 128-byte buffers");
        std::memcpy(memory.mapped, initial.data(), initial.size());
        std::memset(cbuf.mapped, 0, 65536);
        VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        address_info.buffer = memory.buffer;
        uint64_t host = vkGetBufferDeviceAddress(device, &address_info);
        bool zero = std::string(argv[4]) == "zero";
        bool a32 = std::string(argv[4]) == "a32";
        uint64_t delta = zero ? 0 : a32 ? host - 0x2000 :
            std::string(argv[4]) == "negative" ? uint64_t(0) - 0x100000000ull :
            std::string(argv[4]) == "borrow" ? host - 0x1fffffff0ull : 0x100000000ull;
        uint64_t guest = host - delta + 16;
        std::memcpy(cbuf.mapped, &guest, 8);
        const std::array<uint32_t, 4> values{0xff80aabb, 0x76543210, 0xdeadbeef, 0x12345678};
        std::memcpy(static_cast<char*>(cbuf.mapped) + 8, values.data(), sizeof(values));
        VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = 1;
        set_info.pBindings = &binding;
        VkDescriptorSetLayout set_layout{};
        check(vkCreateDescriptorSetLayout(device, &set_info, nullptr, &set_layout));
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &set_layout;
        layout_info.pushConstantRangeCount = zero ? 0 : 1;
        layout_info.pPushConstantRanges = zero ? nullptr : &range;
        VkPipelineLayout layout{};
        check(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout));
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        VkDescriptorPool pool{};
        check(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool));
        VkDescriptorSetAllocateInfo set_allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_allocate.descriptorPool = pool;
        set_allocate.descriptorSetCount = 1;
        set_allocate.pSetLayouts = &set_layout;
        VkDescriptorSet set{};
        check(vkAllocateDescriptorSets(device, &set_allocate, &set));
        VkDescriptorBufferInfo buffer_info{cbuf.buffer, 0, 65536};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &buffer_info;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        auto bytes = read(argv[1]);
        std::vector<uint32_t> words(bytes.size() / 4);
        std::memcpy(words.data(), bytes.data(), bytes.size());
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = bytes.size();
        module_info.pCode = words.data();
        VkShaderModule module{};
        check(vkCreateShaderModule(device, &module_info, nullptr, &module));
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.layout = layout;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        VkPipeline pipeline{};
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline));
        VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_pool_info.queueFamilyIndex = queue_index;
        VkCommandPool command_pool{};
        check(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool));
        VkCommandBufferAllocateInfo command_allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_allocate.commandPool = command_pool;
        command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_allocate.commandBufferCount = 1;
        VkCommandBuffer command{};
        check(vkAllocateCommandBuffers(device, &command_allocate, &command));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(command, &begin));
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        if (!zero) vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &delta);
        vkCmdDispatch(command, argc == 6 ? std::stoul(argv[5]) : 1, 1, 1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        VkQueue queue{};
        vkGetDeviceQueue(device, queue_index, 0, &queue);
        check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(queue));
        bool equal = std::memcmp(memory.mapped, expected.data(), expected.size()) == 0;
        if (!equal) {
            auto actual = static_cast<const unsigned char*>(memory.mapped);
            for (size_t i = 0; i < expected.size(); ++i) {
                if (actual[i] != static_cast<unsigned char>(expected[i]))
                    std::cerr << "byte " << i << ": expected " << unsigned(static_cast<unsigned char>(expected[i])) << ", got " << unsigned(actual[i]) << '\n';
            }
        }
        vkDestroyCommandPool(device, command_pool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyShaderModule(device, module, nullptr);
        vkDestroyDescriptorPool(device, pool, nullptr);
        vkDestroyPipelineLayout(device, layout, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        for (auto buffer : {memory, cbuf}) {
            vkUnmapMemory(device, buffer.memory);
            vkDestroyBuffer(device, buffer.buffer, nullptr);
            vkFreeMemory(device, buffer.memory, nullptr);
        }
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return equal ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
