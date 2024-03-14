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

#include "VulkanDescriptorSet.h"
#include "utils/FixedCapacityVector.h"
#include "utils/Panic.h"
#include "vulkan/VulkanHandles.h"
#include "vulkan/VulkanUtility.h"
#include "vulkan/vulkan_core.h"

#include <string>
#include <vulkan/VulkanConstants.h>
#include <vulkan/VulkanImageUtility.h>
#include <vulkan/VulkanResources.h>

#include <memory>
#include <type_traits>
#include <vector>

#include <math.h>

#include <tsl/robin_map.h>

namespace filament::backend {

namespace {

using Bitmask = VulkanDescriptorSetLayout::Bitmask;

struct BitmaskEqual {
    bool operator()(Bitmask const& k1, Bitmask const& k2) const {
        return k1 == k2;
    }
};

using BitmaskHashFn = utils::hash::MurmurHashFn<Bitmask>;

template <typename T>
std::string printx(T x) {
    std::string o = "0x";
    for (size_t i = 0; i < sizeof(x) * 8; ++i) {
        if (i%16 == 0 && i > 0) o+="-";
        if (x & (1ULL << i)) {
            o+="1";
        } else {
            o+="0";
        }
    }
    return o;
}

// std::string printy1(Bitmask const& bitmask) {
//     std::string ret = "";
//     if (bitmask.ubo) {
//         ret += "[ubo=" + printx(bitmask.ubo) + "]";
//     }
//     if (bitmask.sampler) {
//         ret += " [sampler=" + printx(bitmask.sampler) + "]";
//     }
//     if (bitmask.inputAttachment) {
//         ret += " [inputAttachment=" + printx(bitmask.inputAttachment) + "]";
//     }
//     return ret;
// }

// std::string printy(VulkanDescriptorSetLayout* layout) {
//     std::string ret = printy1(layout->bitmask);
//     if (layout->bitmask.ubo) {
//         ret += "\n\t\tu=";
//         for (auto i : layout->bindings.ubo) {
//             ret += std::to_string(i) + ":";
//         }
//     }
//     if (layout->bitmask.sampler) {
//         ret += "\n\t\ts=";
//         for (auto i : layout->bindings.sampler) {
//             ret += std::to_string(i) + ":";
//         }
//     }
//     if (layout->bitmask.inputAttachment) {
//         ret += "\n\t\ti=";
//         for (auto i : layout->bindings.inputAttachment) {
//             ret += std::to_string(i) + ":";
//         }
//     }
//     return ret;
// }

using ImgUtil = VulkanImageUtility;

// This assumes we have at most 32-bound samplers, 10 UBOs and, 1 input attachment.
// TODO: Safe to remove after [UPCOMING_CHANGE]
constexpr uint8_t MAX_SAMPLER_BINDING = 32;
constexpr uint8_t MAX_UBO_BINDING = 10;
constexpr uint8_t MAX_INPUT_ATTACHMENT_BINDING = 1;
constexpr uint8_t MAX_BINDINGS = MAX_SAMPLER_BINDING + MAX_UBO_BINDING + MAX_INPUT_ATTACHMENT_BINDING;

// This struct can be used to indicate the count of each type of descriptor with respect to a
// layout, or it can be used to indicate the size and capacity of a descriptor pool.
struct DescriptorCount {
    uint32_t ubo = 0;
    uint32_t dynamicUbo = 0;
    uint32_t sampler = 0;
    uint32_t inputAttachment = 0;

    bool operator==(DescriptorCount const& right) const noexcept {
        return ubo == right.ubo && dynamicUbo == right.dynamicUbo &&
                sampler == right.sampler && inputAttachment == right.inputAttachment;
    }

    static inline DescriptorCount fromLayoutBitmask(Bitmask const& mask) {
        return {
            .ubo = countBits(collapseStages(mask.ubo)),
            .dynamicUbo = countBits(collapseStages(mask.dynamicUbo)),
            .sampler = countBits(collapseStages(mask.sampler)),
            .inputAttachment = countBits(collapseStages(mask.inputAttachment)),
        };
    }

    DescriptorCount operator*(uint16_t mult) const noexcept {
        // TODO: check for overflow.

        DescriptorCount ret;
        ret.ubo = ubo * mult;
        ret.dynamicUbo = dynamicUbo * mult;
        ret.sampler = sampler * mult;
        ret.inputAttachment = inputAttachment * mult;
        return ret;
    }

    std::string print() const {
        std::string ret = "";
        ret += "[ubo=" + std::to_string(ubo) + ",dubo=" + std::to_string(dynamicUbo)
            + ",sampler=" + std::to_string(sampler) + ",input=" + std::to_string(inputAttachment)
            + "]";
        return ret;
    }
};

// We create a pool for each layout as defined by the number of descriptors of each type. For
// example, a layout of
// 'A' =>
//   layout(binding = 0, set = 1) uniform {};
//   layout(binding = 1, set = 1) sampler1;
//   layout(binding = 2, set = 1) sampler2;
//
// would be equivalent to
// 'B' =>
//   layout(binding = 1, set = 2) uniform {};
//   layout(binding = 2, set = 2) sampler2;
//   layout(binding = 3, set = 2) sampler3;
//
// TODO: we might do better if we understand the types of unique layouts and can combine them in a
// single pool without too much waste.
class DescriptorPool {
private:
    using Count = DescriptorCount;
public:
    DescriptorPool(VkDevice device, VulkanResourceAllocator* allocator,
            DescriptorCount const& count, uint16_t capacity)
        : mDevice(device),
          mAllocator(allocator),
          mCount(count),
          mCapacity(capacity),
          mSize(0),
          mUnusedCount(0) {
        Count const actual = mCount * capacity;
        utils::slog.e <<"count=" << mCount.print()
                      <<" actual=" << actual.print() << utils::io::endl;
        VkDescriptorPoolSize sizes[4];
        uint8_t npools = 0;
        if (actual.ubo) {
            sizes[npools++] = {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = actual.ubo,
            };
        }
        if (actual.dynamicUbo) {
            sizes[npools++] = {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount = actual.dynamicUbo,
            };
        }
        if (actual.sampler) {
            sizes[npools++] = {
              .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .descriptorCount = actual.sampler,
            };
        }
        if (actual.inputAttachment) {
            sizes[npools++] = {
                .type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                .descriptorCount = actual.inputAttachment,
            };
        }
        VkDescriptorPoolCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = capacity,
            .poolSizeCount = npools,
            .pPoolSizes = sizes,
        };
        vkCreateDescriptorPool(mDevice, &info, VKALLOC, &mPool);
    }

    DescriptorPool(DescriptorPool const&) = delete;
    DescriptorPool& operator=(DescriptorPool const&) = delete;

    ~DescriptorPool() {
        vkDestroyDescriptorPool(mDevice, mPool, VKALLOC);
    }

    uint16_t const& capacity() {
        return mCapacity;
    }

    // A convenience method for checking if this pool can allocate sets for a given layout.
    inline bool canAllocate(VulkanDescriptorSetLayout* layout) {
        return DescriptorCount::fromLayoutBitmask(layout->bitmask) == mCount &&
            (mSize < mCapacity || mUnusedCount > 0);
    }

    Handle<VulkanDescriptorSet> obtainSet(VulkanDescriptorSetLayout* layout) {
        if (UnusedSetMap::iterator itr = mUnused.find(layout->bitmask); itr != mUnused.end()) {
            // If we don't have any unused, then just return an empty handle.
            if (itr->second.empty()) {
                return {};
            }
            std::vector<Handle<VulkanDescriptorSet>>& sets = itr->second;
            auto set = sets.back();
            sets.pop_back();
            mUnusedCount--;
            return set;
        }
        if (mSize + 1 > mCapacity) {
            // This is the equivalent of returning null.
            utils::slog.e << "Maxed out (size=" << mSize << " unused=" << mUnusedCount
                          << ")for type=" << mCount.print() << utils::io::endl;
            return {};
        }
        // Creating a new set
        VkDescriptorSetLayout layouts[1] = {layout->vklayout};
        VkDescriptorSetAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = mPool,
            .descriptorSetCount = 1,
            .pSetLayouts = layouts,
        };
        VkDescriptorSet vkSet;
        UTILS_UNUSED VkResult result = vkAllocateDescriptorSets(mDevice, &allocInfo, &vkSet);
        ASSERT_POSTCONDITION(result == VK_SUCCESS,
                "Failed to allocate descriptor set code=%d size=%d capacity=%d count=%s", result,
                mSize, mCapacity, mCount.print().c_str());
        mSize++;
        return createSet(layout->bitmask, vkSet);
    }

private:
    Handle<VulkanDescriptorSet> createSet(Bitmask const& layoutMask, VkDescriptorSet vkSet) {
        return mAllocator->initHandle<VulkanDescriptorSet>(mAllocator, vkSet,
                [this, layoutMask, vkSet]() {
                    // We are recycling - release the set back into the pool. Note that the
                    // vk handle has not changed, but we need to change the backend handle to allow
                    // for proper refcounting of resources referenced in this set.
                    auto setHandle = createSet(layoutMask, vkSet);
                    if (auto itr = mUnused.find(layoutMask); itr != mUnused.end()) {
                        itr->second.push_back(setHandle);
                    } else {
                        mUnused[layoutMask].push_back(setHandle);
                    }
                    mUnusedCount++;
                });
    }

    VkDevice mDevice;
    VulkanResourceAllocator* mAllocator;
    VkDescriptorPool mPool;
    Count const mCount;
    uint16_t const mCapacity;

    // This tracks that currently the number of in-use descriptor sets.
    uint16_t mSize;
    uint16_t mUnusedCount;

    // This maps a layout ot a list of descriptor sets allocated for that layout.
    using UnusedSetMap = std::unordered_map<Bitmask, std::vector<Handle<VulkanDescriptorSet>>,
            BitmaskHashFn, BitmaskEqual>;
    UnusedSetMap mUnused;
};

class DescriptorInfinitePool {
private:
    static constexpr uint16_t EXPECTED_SET_COUNT = 10;
    static constexpr float SET_COUNT_GROWTH_FACTOR = 1.5;
public:
    DescriptorInfinitePool(VkDevice device, VulkanResourceAllocator* allocator)
        : mDevice(device),
          mResourceAllocator(allocator) {}

    Handle<VulkanDescriptorSet> obtainSet(VulkanDescriptorSetLayout* layout) {
        DescriptorPool* sameTypePool = nullptr;
        for (auto& pool: mPools) {
            if (!pool->canAllocate(layout)) {
                continue;
            }
            if (auto set = pool->obtainSet(layout); set) {
                return set;
            }
            if (!sameTypePool || sameTypePool->capacity() < pool->capacity()) {
                sameTypePool = pool.get();
            }
        }

        uint16_t capacity = EXPECTED_SET_COUNT;
        if (sameTypePool) {
            // Exponentially increase the size of the pool  to ensure we don't hit this too often.
            capacity = std::ceil(sameTypePool->capacity() * SET_COUNT_GROWTH_FACTOR);
        }

        // We need to increase the set of pools by one.
        mPools.push_back(std::make_unique<DescriptorPool>(mDevice, mResourceAllocator,
                DescriptorCount::fromLayoutBitmask(layout->bitmask), capacity));
        auto& pool = mPools.back();
        auto ret = pool->obtainSet(layout);
        assert_invariant(ret && "failed to obtain a set?");
        return ret;
    }

private:
    VkDevice mDevice;
    VulkanResourceAllocator* mResourceAllocator;
    std::vector<std::unique_ptr<DescriptorPool>> mPools;
};

class LayoutCache {
private:
    using Key = Bitmask;

    // Make sure the key is 8-bytes aligned.
    static_assert(sizeof(Key) % 8 == 0);

    using LayoutMap =
            std::unordered_map<Key, Handle<VulkanDescriptorSetLayout>, BitmaskHashFn, BitmaskEqual>;

public:
    explicit LayoutCache(VkDevice device, VulkanResourceAllocator* allocator)
        : mDevice(device),
          mAllocator(allocator) {}

    ~LayoutCache() {
        for (auto [key, layout]: mLayouts) {
            auto layoutPtr = mAllocator->handle_cast<VulkanDescriptorSetLayout*>(layout);
            vkDestroyDescriptorSetLayout(mDevice, layoutPtr->vklayout, VKALLOC);
        }
        mLayouts.clear();
    }

    void destroyLayout(Handle<VulkanDescriptorSetLayout> handle) {
        auto layoutPtr = mAllocator->handle_cast<VulkanDescriptorSetLayout*>(handle);
        for (auto [key, layout]: mLayouts) {
            if (layout == handle) {
                mLayouts.erase(key);
                break;
            }
        }
        vkDestroyDescriptorSetLayout(mDevice, layoutPtr->vklayout, VKALLOC);
    }

    Handle<VulkanDescriptorSetLayout> getLayout(descset::DescriptorSetLayout const& layout) {
        Key key = Bitmask::fromBackendLayout(layout);
        if (auto iter = mLayouts.find(key); iter != mLayouts.end()) {
            return iter->second;
        }

        VkDescriptorSetLayoutBinding toBind[MAX_BINDINGS];
        uint32_t count = 0;

        for (auto const& binding: layout.bindings) {
            VkShaderStageFlags stages = 0;
            VkDescriptorType type;

            if (binding.stageFlags & descset::ShaderStageFlags2::VERTEX) {
                stages |= VK_SHADER_STAGE_VERTEX_BIT;
            }
            if (binding.stageFlags & descset::ShaderStageFlags2::FRAGMENT) {
                stages |= VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            assert_invariant(stages != 0);

            switch (binding.type) {
                case descset::DescriptorType::UNIFORM_BUFFER: {
                    type = binding.flags == descset::DescriptorFlags::DYNAMIC_OFFSET ?
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC :
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    break;
                }
                case descset::DescriptorType::SAMPLER: {
                    type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    break;
                }
                case descset::DescriptorType::INPUT_ATTACHMENT: {
                    type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
                    break;
                }
            }
            toBind[count++] = {
                .binding = binding.binding,
                .descriptorType = type,
                .descriptorCount = 1,
                .stageFlags = stages,
            };
        }

        if (count == 0) {
            return {};
        }

        VkDescriptorSetLayoutCreateInfo dlinfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .bindingCount = count,
            .pBindings = toBind,
        };

        VkDescriptorSetLayout outLayout;
        vkCreateDescriptorSetLayout(mDevice, &dlinfo, VKALLOC, &outLayout);
        return (mLayouts[key] = mAllocator->initHandle<VulkanDescriptorSetLayout>(outLayout, key));
    }

private:
    VkDevice mDevice;
    VulkanResourceAllocator* mAllocator;
    LayoutMap mLayouts;
};

}// anonymous namespace

class VulkanDescriptorSetManager::Impl {
private:
    using GetPipelineLayoutFunction = VulkanDescriptorSetManager::GetPipelineLayoutFunction;

    struct BoundState {
        VkCommandBuffer cmdbuf = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VulkanDescriptorSetList sets;
        VulkanDescriptorSetLayoutList layouts;

        inline bool operator==(BoundState const& b) const {
            if (cmdbuf != b.cmdbuf || pipelineLayout != b.pipelineLayout) {
                return false;
            }
            for (size_t i = 0; i < sets.size(); ++i) {
                if (sets[i] != b.sets[i]) {
                    return false;
                }
                if (layouts[i] != b.layouts[i]) {
                    return false;
                }
            }
            return true;
        }

        inline bool operator!=(BoundState const& b) const { return !(*this == b); }

        inline bool valid() {
            return cmdbuf != VK_NULL_HANDLE;
        }
    };

    static constexpr uint8_t UBO_SET_ID = 0;
    static constexpr uint8_t SAMPLER_SET_ID = 1;
    static constexpr uint8_t INPUT_ATTACHMENT_SET_ID = 2;

public:
    Impl(VkDevice device, VulkanResourceAllocator* allocator)
        : mDevice(device),
          mAllocator(allocator),
          mLayoutCache(device, allocator),
          mDescriptorPool(device, allocator),
          mHaveDynamicUbos(false),
          mResources(allocator) {}

    // This will write/update/bind all of the descriptor set.
    // When bind() is called, that's when the descriptor sets are allocated and then updated. This
    // behavior will change after the [UPCOMING CHANGE] completes.
    VkPipelineLayout bind(VulkanCommandBuffer* commands, VulkanProgram* program,
            GetPipelineLayoutFunction& getPipelineLayoutFn) {
        FVK_SYSTRACE_CONTEXT();
        FVK_SYSTRACE_START("bind");

        VulkanDescriptorSetLayoutList layouts;
        if (auto itr = mLayoutStash.find(program); itr != mLayoutStash.end()) {
            layouts = itr->second;
        } else {
            auto const& layoutDescriptions = program->getLayoutDescriptionList();
            uint8_t count = 0;
            for (auto const& description: layoutDescriptions) {
                layouts[count++] = createLayout(description);
            }
            mLayoutStash[program] = layouts;
        }

        using DescriptorSetVkHandles = utils::FixedCapacityVector<VkDescriptorSet>;
        VulkanDescriptorSetList descSets;
        DescriptorSetVkHandles vkDescSets = DescriptorSetVkHandles::with_capacity(
                VulkanDescriptorSetLayout::UNIQUE_DESCRIPTOR_SET_COUNT);
        VkWriteDescriptorSet descriptorWrites[MAX_BINDINGS];
        VkDescriptorImageInfo inputAttachmentInfo;
        uint32_t nwrites = 0;

        for (uint8_t i = 0; i < VulkanDescriptorSetLayout::UNIQUE_DESCRIPTOR_SET_COUNT; ++i) {
            auto handle = layouts[i];
            if (!handle) {
                assert_invariant(
                        i == INPUT_ATTACHMENT_SET_ID && "Unexpectedly absent descriptor set layout");
                continue;
            }
            auto layout = mAllocator->handle_cast<VulkanDescriptorSetLayout*>(handle);
            if (i == UBO_SET_ID && layout->bitmask.ubo) {
                descSets[i] = mDescriptorPool.obtainSet(layout);
                auto set = mAllocator->handle_cast<VulkanDescriptorSet*>(descSets[i]);
                for (uint8_t binding: layout->bindings.ubo) {
                    auto const& [info, ubo] = mUboMap[binding];
                    descriptorWrites[nwrites++] = {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .pNext = nullptr,
                        .dstSet = set->vkSet,
                        .dstBinding = binding,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .pBufferInfo = ubo  ? &info : &mPlaceHolderBufferInfo,
                    };
                    if (ubo) {
                        set->resources.acquire(ubo);
                    }
                }
                commands->acquire(set);
                vkDescSets.push_back(set->vkSet);
            } else if (i == SAMPLER_SET_ID && layout->bitmask.sampler) {
                descSets[i] = mDescriptorPool.obtainSet(layout);
                auto set = mAllocator->handle_cast<VulkanDescriptorSet*>(descSets[i]);
                for (uint8_t binding: layout->bindings.sampler) {
                    auto const& [info, texture] = mSamplerMap[binding];
                    descriptorWrites[nwrites++] = {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .pNext = nullptr,
                        .dstSet = set->vkSet,
                        .dstBinding = binding,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .pImageInfo = texture ? &info : &mPlaceHolderImageInfo,
                    };
                    if (texture) {
                        set->resources.acquire(texture);
                    }
                }
                commands->acquire(set);
                vkDescSets.push_back(set->vkSet);
            } else if (i == INPUT_ATTACHMENT_SET_ID && layout->bitmask.inputAttachment &&
                    mInputAttachment.texture) {
                descSets[i] = mDescriptorPool.obtainSet(layout);
                auto set = mAllocator->handle_cast<VulkanDescriptorSet*>(descSets[i]);
                set->resources.acquire(mInputAttachment.texture);
                inputAttachmentInfo = {
                    .imageView = mInputAttachment.getImageView(VK_IMAGE_ASPECT_COLOR_BIT),
                    .imageLayout = ImgUtil::getVkLayout(mInputAttachment.getLayout()),
                };
                descriptorWrites[nwrites++] = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = set->vkSet,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                    .pImageInfo = &inputAttachmentInfo,
                };

                commands->acquire(set);
                vkDescSets.push_back(set->vkSet);
            }
        }

        if (nwrites) {
            vkUpdateDescriptorSets(mDevice, nwrites, descriptorWrites, 0, nullptr);
        }

        VkPipelineLayout const pipelineLayout = getPipelineLayoutFn(layouts);
        VkCommandBuffer const cmdbuffer = commands->buffer();

        BoundState state{
            .cmdbuf = cmdbuffer,
            .pipelineLayout = pipelineLayout,
            .sets = descSets,
            .layouts = layouts,
        };

        if (state != mBoundState) {
            vkCmdBindDescriptorSets(cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
                    vkDescSets.size(), vkDescSets.data(), 0, nullptr);
            mBoundState = state;
        }

        // Once bound, the resources are now ref'd in the descriptor set and the some resources can
        // be released and the descriptor set is ref'd by the command buffer.
        for (uint8_t i = 0; i < mSamplerMap.size(); ++i) {
            auto const& [info, texture] = mSamplerMap[i];
            if (texture) {
                mResources.release(texture);
            }
            mSamplerMap[i] = {{}, nullptr};
        }
        mInputAttachment = {};
        mHaveDynamicUbos = false;

        FVK_SYSTRACE_END();
        return pipelineLayout;
    }

    void dynamicBind(VulkanCommandBuffer* commands, Handle<VulkanDescriptorSetLayout> uboLayout) {
        if (!mHaveDynamicUbos) {
            return;
        }
        FVK_SYSTRACE_CONTEXT();
        FVK_SYSTRACE_START("dynamic-bind");

        assert_invariant(mBoundState.valid());
        assert_invariant(commands->buffer() == mBoundState.cmdbuf);

        VkWriteDescriptorSet descriptorWrites[MAX_UBO_BINDING];
        uint8_t nwrites = 0;
        auto layout = mAllocator->handle_cast<VulkanDescriptorSetLayout*>(
                mBoundState.layouts[UBO_SET_ID]);

        // We cannot re-use the previously bound UBO set (because it's bound) so we must get a new
        // set, write to it and bind it.
        // TODO: This is costly, instead just use dynamic UBOs with dynamic offsets.
        auto setHandle = mDescriptorPool.obtainSet(layout);
        auto set = mAllocator->handle_cast<VulkanDescriptorSet*>(setHandle);
        mBoundState.sets[UBO_SET_ID] = setHandle;

        for (uint8_t binding: layout->bindings.ubo) {
            auto const& [info, ubo] = mUboMap[binding];
            descriptorWrites[nwrites++] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = set->vkSet,
                .dstBinding = binding,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = ubo ? &info : &mPlaceHolderBufferInfo,
            };
            if (ubo) {
                set->resources.acquire(ubo);
            }
        }
        if (nwrites == 0) {
            return;
        }
        commands->acquire(set);
        vkUpdateDescriptorSets(mDevice, nwrites, descriptorWrites, 0, nullptr);
        vkCmdBindDescriptorSets(mBoundState.cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                mBoundState.pipelineLayout, 0, 1, &set->vkSet, 0, nullptr);
        mHaveDynamicUbos = false;
        FVK_SYSTRACE_END();
    }

    Handle<VulkanDescriptorSetLayout> createLayout(descset::DescriptorSetLayout const& description) {
        return mLayoutCache.getLayout(description);
    }

    void destroyLayout(Handle<VulkanDescriptorSetLayout> layout) {
        mLayoutCache.destroyLayout(layout);
    }

    // Note that before [UPCOMING CHANGE] arrives, the "update" methods stash state within this
    // class and is not actually working with respect to a descriptor set.
    void updateBuffer(Handle<VulkanDescriptorSet>, uint8_t binding,
            VulkanBufferObject* bufferObject, VkDeviceSize offset, VkDeviceSize size) noexcept {
        VkDescriptorBufferInfo const info {
                .buffer = bufferObject->buffer.getGpuBuffer(),
                .offset = offset,
                .range = size,
        };
        mUboMap[binding] = {info, bufferObject};
        mResources.acquire(bufferObject);

        if (!mHaveDynamicUbos && mBoundState.valid()) {
            mHaveDynamicUbos = true;
        }
    }

    void updateSampler(Handle<VulkanDescriptorSet>, uint8_t binding,
            VulkanTexture* texture, VkSampler sampler) noexcept {
        VkDescriptorImageInfo info {
                .sampler = sampler,
        };
        VkImageSubresourceRange const range = texture->getPrimaryViewRange();
        VkImageViewType const expectedType = texture->getViewType();
        if (any(texture->usage & TextureUsage::DEPTH_ATTACHMENT) &&
                expectedType == VK_IMAGE_VIEW_TYPE_2D) {
            // If the sampler is part of a mipmapped depth texture, where one of the level *can* be
            // an attachment, then the sampler for this texture has the same view properties as a
            // view for an attachment. Therefore, we can use getAttachmentView to get a
            // corresponding VkImageView.
            info.imageView = texture->getAttachmentView(range);
        } else {
            info.imageView = texture->getViewForType(range, expectedType);
        }
        info.imageLayout = ImgUtil::getVkLayout(texture->getPrimaryImageLayout());
        mSamplerMap[binding] = {info, texture};
        mResources.acquire(texture);
    }

    void updateInputAttachment(Handle<VulkanDescriptorSet>,
            VulkanAttachment attachment) noexcept {
        mInputAttachment = attachment;
        mResources.acquire(mInputAttachment.texture);
    }

    void clearBuffer(uint32_t binding) {
        auto const& [info, ubo] = mUboMap[binding];
        if (ubo) {
            mResources.release(ubo);
        }
        mUboMap[binding] = {{}, nullptr};
    }

    void setPlaceHolders(VkSampler sampler, VulkanTexture* texture,
            VulkanBufferObject* bufferObject) noexcept {
        mPlaceHolderBufferInfo = {
            .buffer = bufferObject->buffer.getGpuBuffer(),
            .offset = 0,
            .range = 1,
        };
        mPlaceHolderImageInfo = {
            .sampler = sampler,
            .imageView = texture->getPrimaryImageView(),
            .imageLayout = ImgUtil::getVkLayout(texture->getPrimaryImageLayout()),
        };
    }

    void clearState() noexcept {
        mHaveDynamicUbos = false;
        if (mInputAttachment.texture) {
            mResources.release(mInputAttachment.texture);
        }
        mInputAttachment = {};
    }

private:
    VkDevice mDevice;
    VulkanResourceAllocator* mAllocator;
    LayoutCache mLayoutCache;
    DescriptorInfinitePool mDescriptorPool;
    bool mHaveDynamicUbos;

    std::array<std::pair<VkDescriptorBufferInfo, VulkanBufferObject*>, MAX_UBO_BINDING> mUboMap;
    std::array<std::pair<VkDescriptorImageInfo, VulkanTexture*>, MAX_SAMPLER_BINDING> mSamplerMap;
    VulkanAttachment mInputAttachment;

    VulkanResourceManager mResources;

    VkDescriptorBufferInfo mPlaceHolderBufferInfo;
    VkDescriptorImageInfo mPlaceHolderImageInfo;

    std::unordered_map<VulkanProgram*, VulkanDescriptorSetLayoutList> mLayoutStash;

    BoundState mBoundState;
};

VulkanDescriptorSetManager::VulkanDescriptorSetManager(VkDevice device,
        VulkanResourceAllocator* resourceAllocator)
    : mImpl(new Impl(device, resourceAllocator)) {}

void VulkanDescriptorSetManager::terminate() noexcept {
    assert_invariant(mImpl);
    delete mImpl;
    mImpl = nullptr;
}

// This will write/update/bind all of the descriptor set.
VkPipelineLayout VulkanDescriptorSetManager::bind(VulkanCommandBuffer* commands,
        VulkanProgram* program,
        VulkanDescriptorSetManager::GetPipelineLayoutFunction& getPipelineLayoutFn) {
    return mImpl->bind(commands, program, getPipelineLayoutFn);
}

void VulkanDescriptorSetManager::dynamicBind(VulkanCommandBuffer* commands,
        Handle<VulkanDescriptorSetLayout> uboLayout) {
    mImpl->dynamicBind(commands, uboLayout);
}

Handle<VulkanDescriptorSetLayout> VulkanDescriptorSetManager::createLayout(
        descset::DescriptorSetLayout const& layout) {
    return mImpl->createLayout(layout);
}

void VulkanDescriptorSetManager::destroyLayout(Handle<VulkanDescriptorSetLayout> layout) {
    mImpl->destroyLayout(layout);
}

void VulkanDescriptorSetManager::updateBuffer(Handle<VulkanDescriptorSet> set,
        uint8_t binding, VulkanBufferObject* bufferObject, VkDeviceSize offset,
        VkDeviceSize size) noexcept {
    mImpl->updateBuffer(set, binding, bufferObject, offset, size);
}

void VulkanDescriptorSetManager::updateSampler(Handle<VulkanDescriptorSet> set,
        uint8_t binding, VulkanTexture* texture, VkSampler sampler) noexcept {
    mImpl->updateSampler(set, binding, texture, sampler);
}

void VulkanDescriptorSetManager::updateInputAttachment(Handle<VulkanDescriptorSet> set, VulkanAttachment attachment) noexcept {
    mImpl->updateInputAttachment(set, attachment);
}

void VulkanDescriptorSetManager::clearBuffer(uint32_t bindingIndex) {
    mImpl->clearBuffer(bindingIndex);
}

void VulkanDescriptorSetManager::setPlaceHolders(VkSampler sampler, VulkanTexture* texture,
        VulkanBufferObject* bufferObject) noexcept {
    mImpl->setPlaceHolders(sampler, texture, bufferObject);
}

void VulkanDescriptorSetManager::clearState() noexcept { mImpl->clearState(); }


}// namespace filament::backend
