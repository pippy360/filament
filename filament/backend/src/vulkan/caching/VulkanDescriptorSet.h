/*
 * Copyright (C) 2024 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TNT_FILAMENT_BACKEND_CACHING_VULKANDESCRIPTORSET_H
#define TNT_FILAMENT_BACKEND_CACHING_VULKANDESCRIPTORSET_H

#include <vulkan/VulkanResourceAllocator.h>
#include <vulkan/VulkanTexture.h>
#include <vulkan/VulkanUtility.h>

#include <backend/DriverEnums.h>
#include <backend/Program.h>
#include <backend/TargetBufferInfo.h>

#include <utils/bitset.h>

#include <bluevk/BlueVK.h>
#include <tsl/robin_map.h>

namespace filament::backend {

// We need to make this class public to enable allocation on the HandleAllocator.
struct VulkanDescriptorSet : public VulkanResourceBase {
public:
    // Because we need to recycle descriptor set not used, we allow for a callback that the "Pool"
    // can use to repackage the vk handle.
    using OnRecycle = std::function<void(VulkanDescriptorSet*)>;

    static VulkanDescriptorSet* create(VulkanResourceAllocator* allocator, VkDescriptorSet rawSet,
            VkDescriptorSetLayout layout, OnRecycle&& onRecycleFn);

    // TODO: maybe change to fixed size for performance.
    VulkanAcquireOnlyResourceManager resources;

    VkDescriptorSet const vkSet;

private:
    OnRecycle mOnRecycleFn;

    VulkanDescriptorSet(VulkanResourceAllocator* allocator, VkDescriptorSet rawSet,
            VkDescriptorSetLayout layout, OnRecycle&& onRecycleFn);

    ~VulkanDescriptorSet();

    template <size_t, size_t, size_t>
    friend class HandleAllocator;
};

// Abstraction over the pool and the layout cache.
class VulkanDescriptorSetManager {
public:
    // UBO, samplers, and input attachment
    static constexpr uint8_t DISTINCT_DESCRIPTOR_SET_COUNT = 3;
    static constexpr uint8_t MAX_SUPPORTED_SHADER_STAGE = 2;// Vertex and fragment.

    // static constexpr uint8_t UNIFORM_BINDING_COUNT = Program::UNIFORM_BINDING_COUNT;
    //    static constexpr uint8_t SAMPLER_BINDING_COUNT = Program::SAMPLER_BINDING_COUNT;

    static constexpr uint8_t UNIFORM_BINDING_COUNT = 10;// Program::UNIFORM_BINDING_COUNT;
    static constexpr uint8_t SAMPLER_BINDING_COUNT = 32;// Program::SAMPLER_BINDING_COUNT;

    static_assert(
            sizeof(UniformBufferBitmask) * 8 >= UNIFORM_BINDING_COUNT * MAX_SUPPORTED_SHADER_STAGE);
    static_assert(sizeof(SamplerBitmask) * 8 >= SAMPLER_BINDING_COUNT * MAX_SUPPORTED_SHADER_STAGE);

    static constexpr UniformBufferBitmask UBO_VERTEX_STAGE = 0x1;
    static constexpr UniformBufferBitmask UBO_FRAGMENT_STAGE =
            (0x1ULL << (sizeof(UniformBufferBitmask) * 4));

    static constexpr SamplerBitmask SAMPLER_VERTEX_STAGE = 0x1;
    static constexpr SamplerBitmask SAMPLER_FRAGMENT_STAGE =
            (0x1ULL << (sizeof(SamplerBitmask) * 4));

    static constexpr InputAttachmentBitmask INPUT_ATTACHMENT_VERTEX_STAGE = 0x1;
    static constexpr InputAttachmentBitmask INPUT_ATTACHMENT_FRAGMENT_STAGE =
            (0x1ULL << (sizeof(InputAttachmentBitmask) * 4));

    static constexpr uint8_t UBO_SET_INDEX = 0;
    static constexpr uint8_t SAMPLER_SET_INDEX = 1;
    static constexpr uint8_t INPUT_ATTACHMENT_SET_INDEX = 2;

    using LayoutArray = CappedArray<VkDescriptorSetLayout, DISTINCT_DESCRIPTOR_SET_COUNT>;
    struct SamplerBundle {
        VkDescriptorImageInfo info = {};
        VulkanTexture* texture = nullptr;
        uint8_t binding = 0;
        SamplerBitmask stage = 0;
    };
    using SamplerArray = CappedArray<SamplerBundle, SAMPLER_BINDING_COUNT>;
    using GetPipelineLayoutFunction = std::function<VkPipelineLayout(LayoutArray const&)>;

    VulkanDescriptorSetManager(VkDevice device, VulkanResourceAllocator* resourceAllocator);

    void terminate() noexcept;

    void gc() noexcept;

    // This will write/update/bind all of the descriptor set.
    VkPipelineLayout bind(VulkanCommandBuffer* commands, DescriptorBindingLayout const& layout,
            GetPipelineLayoutFunction& getPipelineFn);

    // This is to "dynamically" bind UBOs that might have offsets changed between pipeline binding
    // and the draw call. We do this because uniforms for primitives that are part of the same
    // renderable can be stored within one uniform buffer. This can be a no-op if there were no
    // range changes between the pipeline bind and the draw call. We will re-use applicable states
    // provided within the bind() call, including the UBO descriptor set layout.
    // TODO: make it a proper dynamic binding when Filament-side descriptor changes are completed.
    void dynamicBind(VulkanCommandBuffer* commands);

    void setUniformBufferObject(uint32_t bindingIndex, VulkanBufferObject* bufferObject,
            VkDeviceSize offset, VkDeviceSize size) noexcept;

    void setSamplers(SamplerArray&& samplers) noexcept;

    void setInputAttachment(VulkanAttachment attachment) noexcept;

    void unsetUniformBufferObject(uint32_t bindingIndex);

    void setPlaceHolders(VkSampler sampler, VulkanTexture* texture,
            VulkanBufferObject* bufferObject) noexcept;

    void clearState() noexcept;

private:
    class Impl;
    Impl* mImpl;
};

}// namespace filament::backend

#endif// TNT_FILAMENT_BACKEND_CACHING_VULKANDESCRIPTORSET_H
