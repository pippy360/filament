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

#include <vulkan/VulkanConstants.h>
#include <vulkan/VulkanImageUtility.h>
#include <vulkan/VulkanResources.h>

#include <type_traits>

namespace filament::backend {

namespace {

using ImgUtil = VulkanImageUtility;

template<typename MaskType>
using LayoutMap = tsl::robin_map<MaskType, VkDescriptorSetLayout>;
using LayoutArray = VulkanDescriptorSetManager::LayoutArray;

constexpr uint8_t EXPECTED_IN_FLIGHT_FRAMES = 3;// Asssume triple buffering

#define PORT_CONSTANT(NameSpace, K) constexpr decltype(NameSpace::K) K = NameSpace::K;

PORT_CONSTANT(VulkanDescriptorSetManager, MAX_SUPPORTED_SHADER_STAGE)
PORT_CONSTANT(VulkanDescriptorSetManager, UBO_VERTEX_STAGE)
PORT_CONSTANT(VulkanDescriptorSetManager, UBO_FRAGMENT_STAGE)

PORT_CONSTANT(VulkanDescriptorSetManager, SAMPLER_VERTEX_STAGE)
PORT_CONSTANT(VulkanDescriptorSetManager, SAMPLER_FRAGMENT_STAGE)

PORT_CONSTANT(VulkanDescriptorSetManager, INPUT_ATTACHMENT_VERTEX_STAGE)
PORT_CONSTANT(VulkanDescriptorSetManager, INPUT_ATTACHMENT_FRAGMENT_STAGE)

PORT_CONSTANT(VulkanDescriptorSetManager, UNIFORM_BINDING_COUNT)
PORT_CONSTANT(VulkanDescriptorSetManager, SAMPLER_BINDING_COUNT)

PORT_CONSTANT(VulkanDescriptorSetManager, DISTINCT_DESCRIPTOR_SET_COUNT)

#undef PORT_CONSTANT

// We only have at most one input attachment, so this bitmask exists only to make the code more
// general.
constexpr decltype(UNIFORM_BINDING_COUNT) INPUT_ATTACHMENT_BINDING_COUNT = 1;

template<typename T>
std::string printx(T x) {
    std::string o = "0x";
    for (size_t i = 0; i < sizeof(x) * 8; ++i) {
        if (i % 16 == 0 && i > 0) o += "-";
        if (((uint64_t) x) & (1ULL << i)) {
            o += "1";
        } else {
            o += "0";
        }
    }
    return o;
}

using DescriptorSetHandles = std::array<VkDescriptorSet, DISTINCT_DESCRIPTOR_SET_COUNT>;

struct SamplerSet {
    using Mask = SamplerBitmask;

    static constexpr Mask VERTEX_STAGE = SAMPLER_VERTEX_STAGE;
    static constexpr Mask FRAGMENT_STAGE = SAMPLER_FRAGMENT_STAGE;

    struct Key {
        uint8_t count;
        uint8_t padding[7];
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        VkSampler sampler[SAMPLER_BINDING_COUNT];
        VkImageView imageView[SAMPLER_BINDING_COUNT];
        VkImageLayout imageLayout[SAMPLER_BINDING_COUNT];
    };

    struct Equal {
        bool operator()(Key const& k1, Key const& k2) const {
            if (k1.count != k2.count) return false;
            if (k1.layout != k2.layout) return false;

            for (uint8_t i = 0, bitCount = k1.count; i < bitCount; ++i) {
                if (k1.sampler[i] != k2.sampler[i] || k1.imageView[i] != k2.imageView[i] ||
                        k1.imageLayout[i] != k2.imageLayout[i]) {
                    return false;
                }
            }
            return true;
        }
    };
    using HashFn = utils::hash::MurmurHashFn<Key>;
    using Cache = std::unordered_map<Key, VulkanDescriptorSet*, HashFn, Equal>;
};

struct UBOSet {
    using Mask = UniformBufferBitmask;

    static constexpr Mask VERTEX_STAGE = UBO_VERTEX_STAGE;
    static constexpr Mask FRAGMENT_STAGE = UBO_FRAGMENT_STAGE;

    struct Key {
        uint8_t count;
        uint8_t padding[7];
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        VkBuffer buffers[UNIFORM_BINDING_COUNT];
        VkDeviceSize offsets[UNIFORM_BINDING_COUNT];
        VkDeviceSize sizes[UNIFORM_BINDING_COUNT];
    };

    struct Equal {
        bool operator()(Key const& k1, Key const& k2) const {
            if (k1.count != k2.count) return false;
            if (k1.layout != k2.layout) return false;

            for (uint8_t i = 0, bitCount = k1.count; i < bitCount; ++i) {
                if (k1.buffers[i] != k2.buffers[i] || k1.offsets[i] != k2.offsets[i] ||
                        k1.sizes[i] != k2.sizes[i]) {
                    return false;
                }
            }
            return true;
        }
    };
    using HashFn = utils::hash::MurmurHashFn<Key>;
    using Cache = std::unordered_map<Key, VulkanDescriptorSet*, HashFn, Equal>;
};

struct InputAttachmentSet {
    using Mask = InputAttachmentBitmask;

    static constexpr Mask VERTEX_STAGE = INPUT_ATTACHMENT_VERTEX_STAGE;
    static constexpr Mask FRAGMENT_STAGE = INPUT_ATTACHMENT_FRAGMENT_STAGE;

    struct Key {
        // This count should be fixed.
        uint8_t count = INPUT_ATTACHMENT_BINDING_COUNT;
        uint8_t padding[7];
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };
    static_assert(sizeof(Key) == 32);

    struct Equal {
        bool operator()(Key const& k1, Key const& k2) const {
            return k1.layout == k2.layout && k1.view == k2.view && k1.imageLayout == k2.imageLayout;
        }
    };
    using HashFn = utils::hash::MurmurHashFn<Key>;
    using Cache = std::unordered_map<Key, VulkanDescriptorSet*, HashFn, Equal>;
};

constexpr std::array<uint32_t, DISTINCT_DESCRIPTOR_SET_COUNT> BINDING_COUNTS = {
    UNIFORM_BINDING_COUNT,
    SAMPLER_BINDING_COUNT,
    INPUT_ATTACHMENT_BINDING_COUNT,
};

constexpr std::array<char const*, DISTINCT_DESCRIPTOR_SET_COUNT> DESCRIPTOR_TYPE_NAME = {
    "ubo",
    "sampler",
    "input attachment",
};

constexpr std::array<VkDescriptorType, DISTINCT_DESCRIPTOR_SET_COUNT> DESCRIPTOR_TYPES = {
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
};

template<typename TYPE, typename OutType>
constexpr OutType chooseGivenType(
        std::array<OutType, DISTINCT_DESCRIPTOR_SET_COUNT> const& values) {
    if constexpr (std::is_same_v<TYPE, UBOSet>) {
        return values[0];
    }
    if constexpr (std::is_same_v<TYPE, SamplerSet>) {
        return values[1];
    }
    if constexpr (std::is_same_v<TYPE, InputAttachmentSet>) {
        return values[2];
    }
}

template<typename TYPE>
class DescriptorPool {
private:
    static constexpr uint32_t MAX_SET_COUNT = EXPECTED_IN_FLIGHT_FRAMES * 10;
    static constexpr uint32_t MAX_DESCRIPTOR_COUNT =
            chooseGivenType<TYPE>(BINDING_COUNTS) * MAX_SET_COUNT;

public:
    DescriptorPool()
        : mDevice(VK_NULL_HANDLE),
          mResourceAllocator(nullptr) {}

    DescriptorPool(VkDevice device, VulkanResourceAllocator* allocator)
        : mDevice(device),
          mResourceAllocator(allocator) {
        VkDescriptorPoolSize sizes[1] = {
            {
                .type = chooseGivenType<TYPE>(DESCRIPTOR_TYPES),
                .descriptorCount = MAX_DESCRIPTOR_COUNT,
            },
        };
        VkDescriptorPoolCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = MAX_SET_COUNT,
            .poolSizeCount = 1,
            .pPoolSizes = sizes,
        };
        vkCreateDescriptorPool(mDevice, &info, VKALLOC, &mPool);
    }

    DescriptorPool(DescriptorPool const&) = delete;
    DescriptorPool& operator=(DescriptorPool const&) = delete;

    ~DescriptorPool() { vkDestroyDescriptorPool(mDevice, mPool, VKALLOC); }

    VulkanDescriptorSet* obtainSet(uint8_t descriptorCount, VkDescriptorSetLayout layout) {
        if (auto unused = mUnused.find(layout);
                unused != mUnused.end() && !unused->second.empty()) {
            auto& unusedList = unused->second;
            auto set = unusedList.back();
            unusedList.pop_back();
            mDescriptorCount += descriptorCount;
            mSetCount++;
            return set;
        }

        if (mActualDescriptors + descriptorCount > MAX_DESCRIPTOR_COUNT ||
                mActualSets + 1 > MAX_SET_COUNT) {
            return nullptr;
        }

        // Creating a new set
        VkDescriptorSetLayout layouts[1] = {layout};
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
                "Cannot allocate descriptor set. error=%d allocated=%d request=%d max=%d"
                " set-alloc=%d set-max=%d actual-set=%d actual=%d",
                (int) result, mDescriptorCount, descriptorCount, MAX_DESCRIPTOR_COUNT, mSetCount,
                             mActualSets, mActualDescriptors, MAX_SET_COUNT);

        mActualSets++;
        mActualDescriptors += descriptorCount;

        mDescriptorCount += descriptorCount;
        auto ret = createSet(layout, vkSet, descriptorCount);
        mSetCount++;
        return ret;
    }

private:
    inline VulkanDescriptorSet* createSet(VkDescriptorSetLayout layout, VkDescriptorSet vkSet,
            uint8_t descriptorCount) {
        if (mUnused.find(layout) == mUnused.end()) {
            mUnused[layout] = {};
        }
        return VulkanDescriptorSet::create(mResourceAllocator, vkSet, layout,
                [this, layout, descriptorCount](VulkanDescriptorSet* set) {
                    auto const vkSet = set->vkSet;
                    auto listIter = mUnused.find(layout);
                    assert_invariant(listIter != mUnused.end());
                    auto& unusedList = listIter->second;

                    assert_invariant(std::find(unusedList.begin(), unusedList.end(), set) ==
                                     unusedList.end());

                    // We are recycling - release the vk resource back into the pool. Note that the
                    // vk handle has not changed, but we need to change the backend handle to allow
                    // for proper refcounting of resources referenced in this set.
                    unusedList.push_back(createSet(layout, vkSet, descriptorCount));
                    mDescriptorCount -= descriptorCount;
                    mSetCount--;
                });
    }

    VkDevice mDevice;
    VulkanResourceAllocator* mResourceAllocator;
    VkDescriptorPool mPool;

    uint16_t mDescriptorCount = 0;
    uint8_t mSetCount = 0;

    uint32_t mActualSets = 0;
    uint32_t mActualDescriptors = 0;    

    // This maps a layout ot a list of descriptor sets allocated for that layout.
    std::unordered_map<VkDescriptorSetLayout, std::vector<VulkanDescriptorSet*>> mUnused;
};

template<typename TYPE>
class DescriptorInfinitePool {
public:
    DescriptorInfinitePool(VkDevice device, VulkanResourceAllocator* allocator)
        : mDevice(device),
          mResourceAllocator(allocator) {}

    ~DescriptorInfinitePool() {
        for (auto pool: mPools) {
            delete pool;
        }
    }

    VulkanDescriptorSet* obtainSet(uint8_t descriptorCount, VkDescriptorSetLayout layout) {
        for (auto pool: mPools) {
            if (auto set = pool->obtainSet(descriptorCount, layout); set) {
                return set;
            }
        }
        // We need to increase the number of pools
        mPools.push_back(new DescriptorPool<TYPE>{mDevice, mResourceAllocator});
        auto pool = mPools.back();
        return pool->obtainSet(descriptorCount, layout);
    }

private:
    VkDevice mDevice;
    VulkanResourceAllocator* mResourceAllocator;
    std::vector<DescriptorPool<TYPE>*> mPools;
};

// This holds a cache of descriptor sets of a TYPE (UBO or Sampler). The Key is defined as the
// layout and the content (for example specific samplers).
template<typename TYPE>
class DescriptorSetCache {
private:
    using Key = typename TYPE::Key;

    static constexpr char const* TYPE_NAME = chooseGivenType<TYPE>(DESCRIPTOR_TYPE_NAME);

public:
    DescriptorSetCache(VkDevice device, VulkanResourceAllocator* allocator)
        : mPool(device, allocator),
          mResourceManager(allocator) {}

    std::pair<VulkanDescriptorSet*, bool> obtainSet(Key const& key) {
        /*
        mAge++;
        if (auto result = mCache.find(key); result != mCache.end()) {
            VulkanDescriptorSet* set = result->second;
            auto const oldAge = mReverseHistory[set];
            mHistory.erase(oldAge);
            mHistory[mAge] = set;
            mReverseHistory[set] = mAge;
            utils::slog.e << TYPE_NAME << "-> [cached]=" << mHistory.size() << utils::io::endl;
            return {set, true};
        }
        */

        VulkanDescriptorSet* set = mPool.obtainSet(key.count, key.layout);
        /*
        mCache[key] = set;
        mReverseCache[set] = key;
        mHistory[mAge] = set;
        mReverseHistory[set] = mAge;
        mResourceManager.acquire(set);
        */

//        utils::slog.e << TYPE_NAME << "-> [not cached]=" << mHistory.size() << utils::io::endl;
        return {set, false};
    }

    // We need to periodically purge the descriptor sets so that we're not holding on to resources
    // unnecessarily. The strategy for purging needs to be examined more.
    void gc() noexcept {
        /*
        constexpr uint32_t ALLOWED_ENTRIES = EXPECTED_IN_FLIGHT_FRAMES * 10;
        int32_t toCutCount = ((int32_t) mHistory.size()) - ALLOWED_ENTRIES;

        if (toCutCount <= 0) {
            return;
        }

        std::vector<uint64_t> remove;
//        for (auto entry = mHistory.begin(); entry != mHistory.end() && toCutCount > 0;
//                entry++, toCutCount--) {
        for (auto entry = mHistory.begin(); entry != mHistory.end(); entry++) {
            auto const& set = entry->second;
            Key const& key = mReverseCache[set];
            mCache.erase(key);
            mReverseCache.erase(set);
            mReverseHistory.erase(set);

            remove.push_back(entry->first);
            mResourceManager.release(set);
        }
        for (auto removeAge: remove) {
            mHistory.erase(removeAge);
        }
        */
    }

private:
    DescriptorInfinitePool<TYPE> mPool;

    // TODO: combine some of these data structures
    typename TYPE::Cache mCache;
    std::unordered_map<VulkanDescriptorSet*, Key> mReverseCache;

    // Use the ordering for purging if needed;
    std::map<uint64_t, VulkanDescriptorSet*> mHistory;
    std::unordered_map<VulkanDescriptorSet*, uint64_t> mReverseHistory;
    uint64_t mAge;

    // Note that instead of owning the resources (i.e. descriptor set) in the pools, we keep them
    // here since all the allocated sets have to pass through this class, and this class has better
    // knowledge to make decisions about gc.
    VulkanResourceManager mResourceManager;
};

template<typename SetType>
class LayoutCache {
private:
    using MaskType = typename SetType::Mask;

public:
    explicit LayoutCache(VkDevice device)
        : mDevice(device) {}

    ~LayoutCache() {
        for (auto [mask, layout]: mLayouts) {
            vkDestroyDescriptorSetLayout(mDevice, layout, VKALLOC);
        }
        mLayouts.clear();
    }

    VkDescriptorSetLayout getLayout(MaskType stageMask) noexcept {
        if (stageMask == 0) {
            return VK_NULL_HANDLE;
        }
        if (auto iter = mLayouts.find(stageMask); iter != mLayouts.end()) {
            return iter->second;
        }
        VkDescriptorType descriptorType = chooseGivenType<SetType>(DESCRIPTOR_TYPES);

        constexpr decltype(SetType::VERTEX_STAGE) VERTEX_STAGE = SetType::VERTEX_STAGE;
        constexpr decltype(SetType::FRAGMENT_STAGE) FRAGMENT_STAGE = SetType::FRAGMENT_STAGE;
        constexpr uint8_t MAX_BINDINGS =
                chooseGivenType<SetType>(BINDING_COUNTS) * MAX_SUPPORTED_SHADER_STAGE;

        VkDescriptorSetLayoutBinding toBind[MAX_BINDINGS];
        uint32_t count = 0;
        for (uint8_t i = 0, maxBindings = sizeof(stageMask) * 4; i < maxBindings; i++) {
            VkShaderStageFlags stages = 0;
            if (stageMask & (VERTEX_STAGE << i)) {
                stages |= VK_SHADER_STAGE_VERTEX_BIT;
            }
            if (stageMask & (FRAGMENT_STAGE << i)) {
                stages |= VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            if (stages == 0) {
                continue;
            }
            auto& bindInfo = toBind[count++];
            bindInfo = {
                .binding = i,
                .descriptorType = descriptorType,
                .descriptorCount = 1,
                .stageFlags = stages,
            };
        }
        VkDescriptorSetLayoutCreateInfo dlinfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .bindingCount = count,
            .pBindings = toBind,
        };

        VkDescriptorSetLayout handle;
        vkCreateDescriptorSetLayout(mDevice, &dlinfo, VKALLOC, &handle);
        mLayouts[stageMask] = handle;
        return handle;
    }

private:
    VkDevice mDevice;
    LayoutMap<MaskType> mLayouts;
};

}// anonymous namespace

VulkanDescriptorSet::VulkanDescriptorSet(VulkanResourceAllocator* allocator, VkDescriptorSet rawSet,
        VkDescriptorSetLayout layout, OnRecycle&& onRecycleFn)
    : VulkanResource(VulkanResourceType::DESCRIPTOR_SET),
      resources(allocator),
      vkSet(rawSet),
      mOnRecycleFn(std::move(onRecycleFn)) {
}

VulkanDescriptorSet::~VulkanDescriptorSet() {
    if (mOnRecycleFn) {
        mOnRecycleFn(this);
    }
}

VulkanDescriptorSet* VulkanDescriptorSet::create(VulkanResourceAllocator* allocator,
        VkDescriptorSet rawSet, VkDescriptorSetLayout layout, OnRecycle&& onRecycleFn) {
    auto handle = allocator->allocHandle<VulkanDescriptorSet>();
    auto set = allocator->construct<VulkanDescriptorSet>(handle, allocator, rawSet, layout,
            std::move(onRecycleFn));
    return set;
}

class VulkanDescriptorSetManager::Impl {
public:
    Impl(VkDevice device, VulkanResourceAllocator* allocator)
        : mDevice(device),
          mUBOLayoutCache(device),
          mUBOCache(device, allocator),
          mSamplerLayoutCache(device),
          mSamplerCache(device, allocator),
          mInputAttachmentLayoutCache(device),
          mInputAttachmentCache(device, allocator),
          mHaveDynamicUbos(false),
          mResources(allocator) {}

    void setUniformBufferObject(uint32_t bindingIndex, VulkanBufferObject* bufferObject,
            VkDeviceSize offset, VkDeviceSize size) noexcept {
        if (!mPreviousBoundState.uninitialized() && !mHaveDynamicUbos) {
            assert_invariant(mUbos.find(bindingIndex) != mUbos.end());
            mHaveDynamicUbos = true;
        }

        if (mUbos.find(bindingIndex) != mUbos.end()) {
            if (mUbos[bindingIndex].bufferObj != bufferObject) {
                mResources.release(mUbos[bindingIndex].bufferObj);
            }
        }
        mUbos[bindingIndex] = {bufferObject, offset, size};

        // Between "set" and "commit", we need to ref the buffer object to avoid it being collected.
        mResources.acquire(bufferObject);
    }

    void setSamplers(SamplerArray&& samplers) noexcept {
        mSamplers = std::move(samplers);
        for (auto const& sampler: mSamplers) {
            mResources.acquire(sampler.texture);
        }
    }

    void setInputAttachment(VulkanAttachment attachment) noexcept {
        mInputAttachment = attachment;
        if (attachment.texture) {
            mResources.acquire(attachment.texture);
        }
    }

    void gc() noexcept {
        mUBOCache.gc();
        mSamplerCache.gc();
        mInputAttachmentCache.gc();
    }

    // This will write/update all of the descriptor set (and create a set if a one of the same
    // layout is not available).
    VkPipelineLayout bind(VulkanCommandBuffer* commands, DescriptorBindingLayout const& layout,
            GetPipelineLayoutFunction& getPipelineLayoutFn) {

//        utils::slog.e <<"-----------binding -----------------" << utils::io::endl;

        LayoutArray layouts;
        DescriptorSetHandles descSets{VK_NULL_HANDLE};
        uint8_t descSetCount = 0;

        VkWriteDescriptorSet descriptorWrites[UNIFORM_BINDING_COUNT + SAMPLER_BINDING_COUNT +
                                              INPUT_ATTACHMENT_BINDING_COUNT];
        VkDescriptorBufferInfo uboWrite[UNIFORM_BINDING_COUNT];
        VkDescriptorImageInfo inputAttachmentInfo;
        VkDescriptorImageInfo placeHolderSamplerInfo;
        uint32_t nwrites = 0;

        auto const& uboMask = layout.ubo;
        auto const& samplerMask = layout.sampler;

        if (uboMask) {
            auto [descriptorSet, vkset, vklayout, writes] =
                writeUbos(uboMask, descriptorWrites, uboWrite, nwrites);

            nwrites = writes;
            descSets[descSetCount++] = vkset;
            layouts.insert(vklayout);
            commands->acquire(descriptorSet);

//            utils::slog.e << "ubo=" << printx(uboMask) << " " << layouts[0] << utils::io::endl;
        }

        if (samplerMask) {
            SamplerSet::Key samplerKey = {};
            struct SamplerAux {
                uint8_t binding;
                VulkanTexture* texture;
                VkDescriptorImageInfo const* info;
            };
            std::array<SamplerAux, SAMPLER_BINDING_COUNT> samplerAux;
            auto& samplerCount = samplerKey.count;

            auto const setSampler = [&samplerKey, &samplerAux](uint8_t samplerCount,
                                            SamplerBundle const& sampler) {
                auto& aux = samplerAux[samplerCount];
                aux.binding = sampler.binding;
                aux.texture = sampler.texture;

                // Note that we're pointing to data in mSamplers, and so sampler needs to be a
                // const-ref and mSamplers needs to exist until the descriptor write completes.
                aux.info = &sampler.info;

                samplerKey.sampler[samplerCount] = sampler.info.sampler;
                samplerKey.imageView[samplerCount] = sampler.info.imageView;
                samplerKey.imageLayout[samplerCount] = sampler.info.imageLayout;
            };

            SamplerBitmask usedSamplerMask = 0;
            for (auto const& sampler: mSamplers) {
                auto const binding = sampler.binding;
                auto const stages =
                        samplerMask & ((SAMPLER_VERTEX_STAGE | SAMPLER_FRAGMENT_STAGE) << binding);
                if (stages == 0) {
                    continue;
                }
                usedSamplerMask |= stages;
                setSampler(samplerCount++, sampler);
            }

            // Note that this needs to be allocated in the function block (due to us using it's
            // children struct in the vk calls).
            SamplerBundle tmpSampler;
            if (SamplerBitmask leftoverSampler = (~usedSamplerMask) & samplerMask;
                    leftoverSampler) {
                auto layout = ImgUtil::getVkLayout(mPlaceHolderTexture->getPrimaryImageLayout());
                placeHolderSamplerInfo = {
                        .sampler = mPlaceHolderSampler,
                        .imageView = mPlaceHolderTexture->getPrimaryImageView(),
                        .imageLayout = layout,
                };
                tmpSampler = {
                        .info = placeHolderSamplerInfo,
                        .texture = mPlaceHolderTexture,
                };

                for (uint8_t binding = 0; binding < SAMPLER_BINDING_COUNT && leftoverSampler > 0;
                        leftoverSampler /= 2, binding++) {
                    if (leftoverSampler & (SAMPLER_VERTEX_STAGE | SAMPLER_FRAGMENT_STAGE)) {
                        tmpSampler.binding = binding;
                        tmpSampler.stage =
                                samplerMask &
                                ((SAMPLER_VERTEX_STAGE | SAMPLER_FRAGMENT_STAGE) << binding);
                        setSampler(samplerCount++, tmpSampler);
                    }
                }
            }

            samplerKey.layout = mSamplerLayoutCache.getLayout(samplerMask);
            layouts.insert(samplerKey.layout);

            auto [descriptorSet, cached] = mSamplerCache.obtainSet(samplerKey);
            descSets[descSetCount++] = descriptorSet->vkSet;

            if (!cached) {
                for (uint8_t i = 0; i < samplerCount; ++i) {
                    auto const& binding = samplerAux[i].binding;
                    descriptorSet->resources.acquire(samplerAux[i].texture);
                    VkWriteDescriptorSet& descriptorWrite = descriptorWrites[nwrites++];
                    descriptorWrite = {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .pNext = nullptr,
                        .dstSet = descriptorSet->vkSet,
                        .dstBinding = binding,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .pImageInfo = samplerAux[i].info,
                    };
                }
            }
            commands->acquire(descriptorSet);

//            utils::slog.e << "sampler=" << printx(samplerMask) << " " << layouts[layouts.size()-1] << utils::io::endl;
        }

        // For subpass, we will always assign a binding in the layout if it is in the shaders.
        if (layout.inputAttachment) {
            InputAttachmentSet::Key inputAttachmentKey;
            inputAttachmentKey.layout =
                    mInputAttachmentLayoutCache.getLayout(layout.inputAttachment);
            layouts.insert(inputAttachmentKey.layout);

            auto [descriptorSet, cached] = mInputAttachmentCache.obtainSet(inputAttachmentKey);
            descSets[descSetCount++] = descriptorSet->vkSet;

            // However, we only need to write to it if there's actually an input attachment for this
            // subpass.
            if (mInputAttachment.texture) {

                inputAttachmentInfo = {
                    .imageView = mInputAttachment.getImageView(VK_IMAGE_ASPECT_COLOR_BIT),
                    .imageLayout = ImgUtil::getVkLayout(mInputAttachment.getLayout()),
                };
                descriptorWrites[nwrites++] = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = descriptorSet->vkSet,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                    .pImageInfo = &inputAttachmentInfo,
                };
            }
            commands->acquire(descriptorSet);

//            utils::slog.e << "input=" << printx(layout.inputAttachment) << " " << layouts[layouts.size()-1] << utils::io::endl;
        }

        if (nwrites) {
            vkUpdateDescriptorSets(mDevice, nwrites, descriptorWrites, 0, nullptr);
        }

        VkPipelineLayout const pipelineLayout = getPipelineLayoutFn(layouts);

//        utils::slog.e <<"pipeline=" <<pipelineLayout << utils::io::endl;

        VkCommandBuffer const cmdbuffer = commands->buffer();

        BoundState state {
            .cmdbuf = cmdbuffer,
            .pipelineLayout = pipelineLayout,
            .handles = descSets,
            .uboMask = uboMask,
        };

        if (state != mPreviousBoundState) {
            vkCmdBindDescriptorSets(cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
                    descSetCount, descSets.data(), 0, nullptr);
            mPreviousBoundState = state;
        }

        // Once bound, the resources are now ref'd in the descriptor set and the some resources can
        // be released and the descriptor set is ref'd by the command buffer.
        for (auto const& sampler: mSamplers) {
            mResources.release(sampler.texture);
        }
        mSamplers.clear();
        mInputAttachment = {};

        return pipelineLayout;
    }

    void dynamicBind(VulkanCommandBuffer* commands) {

        if (!mHaveDynamicUbos) {
            return;
        }

        assert_invariant(!mPreviousBoundState.uninitialized());

        VkWriteDescriptorSet descriptorWrites[UNIFORM_BINDING_COUNT];
        VkDescriptorBufferInfo uboWrite[UNIFORM_BINDING_COUNT];

        auto [descriptorSet, vkset, vklayout, nwrites] =
                writeUbos(mPreviousBoundState.uboMask, descriptorWrites, uboWrite, 0);

        if (nwrites == 0) {
            return;
        }

        commands->acquire(descriptorSet);

        vkUpdateDescriptorSets(mDevice, nwrites, descriptorWrites, 0, nullptr);

        // Only bind the UBO set
        vkCmdBindDescriptorSets(mPreviousBoundState.cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                mPreviousBoundState.pipelineLayout, 0, 1, &vkset, 0, nullptr);

        mHaveDynamicUbos = false;
    }

    void unsetUniformBufferObject(uint32_t bindingIndex) {
        if (auto itr = mUbos.find(bindingIndex); itr != mUbos.end()) {
            mResources.release(itr->second.bufferObj);
            mUbos.erase(bindingIndex);
        }
    }

    void clearState() noexcept {
        mPreviousBoundState = {};
        mInputAttachment = {};
        mHaveDynamicUbos = false;
    }

    void setPlaceHolders(VkSampler sampler, VulkanTexture* texture, VulkanBufferObject* bufferObject) noexcept {
        mPlaceHolderObject = bufferObject;
        mPlaceHolderTexture = texture;
        mPlaceHolderSampler = sampler;
    }

private:
    inline std::tuple<VulkanDescriptorSet*, VkDescriptorSet, VkDescriptorSetLayout, uint32_t>
    writeUbos(UniformBufferBitmask const& uboMask, VkWriteDescriptorSet* descriptorWrites,
            VkDescriptorBufferInfo* uboWrite, uint32_t nwrites) {
        UBOSet::Key uboKey = {};
        struct UBOAux {
            uint8_t binding;
            VulkanBufferObject* buffer;
        };
        std::array<UBOAux, UNIFORM_BINDING_COUNT> uboAux;
        auto& uboCount = uboKey.count;
        auto const setUbo = [&uboKey, &uboAux](uint8_t uboCount, uint8_t binding,
                                    VulkanBufferObject* bo, VkDeviceSize const& offset,
                                    VkDeviceSize const& size) {
            auto& aux = uboAux[uboCount];
            aux.binding = binding;
            aux.buffer = bo;

            uboKey.buffers[uboCount] = bo->buffer.getGpuBuffer();
            uboKey.offsets[uboCount] = offset;
            uboKey.sizes[uboCount] = size;
        };

        UniformBufferBitmask usedUboMask = 0;
        for (auto const& [binding, ubo]: mUbos) {
            auto const stages = uboMask & ((UBO_VERTEX_STAGE | UBO_FRAGMENT_STAGE) << binding);
            if (stages == 0) {
                continue;
            }
            usedUboMask |= stages;
            setUbo(uboCount++, binding, ubo.bufferObj, ubo.offset, ubo.size);
        }

        // These are ubos that are specified in the shader but have not been updated.
        UniformBufferBitmask leftoverUbo = (~usedUboMask) & uboMask;
        for (uint8_t binding = 0; binding < UNIFORM_BINDING_COUNT && leftoverUbo > 0;
                leftoverUbo /= 2, binding++) {
            if (leftoverUbo & (UBO_VERTEX_STAGE | UBO_FRAGMENT_STAGE)) {
                setUbo(uboCount++, binding, mPlaceHolderObject, 0, 1);
            }
        }

        uboKey.layout = mUBOLayoutCache.getLayout(uboMask);

        auto [descriptorSet, cached] = mUBOCache.obtainSet(uboKey);

        // We need to write to the descriptor set since it wasn't cached.
        if (!cached) {
            for (uint8_t i = 0; i < uboCount; ++i) {
                // If it wasn't cached before, we need to ref the resources that it touches.
                auto const buffer = uboAux[i].buffer;
                descriptorSet->resources.acquire(buffer);

                auto const& binding = uboAux[i].binding;
                VkWriteDescriptorSet& descriptorWrite = descriptorWrites[nwrites++];
                auto& writeInfo = uboWrite[i];
                writeInfo = {
                        .buffer = uboKey.buffers[i],
                        .offset = uboKey.offsets[i],
                        .range = uboKey.sizes[i],
                };
                descriptorWrite = {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .pNext = nullptr,
                        .dstSet = descriptorSet->vkSet,
                        .dstBinding = binding,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .pBufferInfo = &writeInfo,
                };
            }
        }
        
        return {descriptorSet, descriptorSet->vkSet, uboKey.layout, nwrites};
    }

private:
    struct BoundState {
        VkCommandBuffer cmdbuf = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        DescriptorSetHandles handles {VK_NULL_HANDLE};
        UniformBufferBitmask uboMask;

        inline bool operator==(BoundState const& b) const {
            return handles == b.handles && cmdbuf == b.cmdbuf && pipelineLayout == b.pipelineLayout;
        }

        inline bool operator!=(BoundState const& b) const { return !(*this == b); }


        inline bool uninitialized() {
            return cmdbuf == VK_NULL_HANDLE;
        }
    };

    VkDevice mDevice;

    LayoutCache<UBOSet> mUBOLayoutCache;
    DescriptorSetCache<UBOSet> mUBOCache;

    LayoutCache<SamplerSet> mSamplerLayoutCache;
    DescriptorSetCache<SamplerSet> mSamplerCache;

    LayoutCache<InputAttachmentSet> mInputAttachmentLayoutCache;
    DescriptorSetCache<InputAttachmentSet> mInputAttachmentCache;

    // Current states (kept until bind is called)
    struct UBOBundle {
        VulkanBufferObject* bufferObj;
        VkDeviceSize offset;
        VkDeviceSize size;
    };

    std::unordered_map<uint32_t, UBOBundle> mUbos;

    // We use this to denote the dyanmic ranges on UBOs that might be set between a pipeline bind
    // and draw.
    bool mHaveDynamicUbos;
    SamplerArray mSamplers;
    VulkanAttachment mInputAttachment = {};

    VulkanResourceManager mResources;

    VkSampler mPlaceHolderSampler;
    VulkanTexture* mPlaceHolderTexture;
    VulkanBufferObject* mPlaceHolderObject;

    BoundState mPreviousBoundState;
};

VulkanDescriptorSetManager::VulkanDescriptorSetManager(VkDevice device,
        VulkanResourceAllocator* resourceAllocator)
    : mImpl(new Impl(device, resourceAllocator)) {}

void VulkanDescriptorSetManager::terminate() noexcept {
    assert_invariant(mImpl);
    delete mImpl;
    mImpl = nullptr;
}

void VulkanDescriptorSetManager::gc() noexcept { mImpl->gc(); }

VkPipelineLayout VulkanDescriptorSetManager::bind(VulkanCommandBuffer* commands,
        DescriptorBindingLayout const& layout, GetPipelineLayoutFunction& getPipelineLayoutFn) {
    return mImpl->bind(commands, layout, getPipelineLayoutFn);
}

void VulkanDescriptorSetManager::dynamicBind(VulkanCommandBuffer* commands) {
    mImpl->dynamicBind(commands);
}

void VulkanDescriptorSetManager::setUniformBufferObject(uint32_t bindingIndex,
        VulkanBufferObject* bufferObject, VkDeviceSize offset, VkDeviceSize size) noexcept {
    mImpl->setUniformBufferObject(bindingIndex, bufferObject, offset, size);
}

void VulkanDescriptorSetManager::setSamplers(SamplerArray&& samplers) noexcept {
    mImpl->setSamplers(std::move(samplers));
}

void VulkanDescriptorSetManager::setInputAttachment(VulkanAttachment attachment) noexcept {
    mImpl->setInputAttachment(attachment);
}

void VulkanDescriptorSetManager::setPlaceHolders(VkSampler sampler,
        VulkanTexture* texture,  VulkanBufferObject* bufferObject) noexcept {
    mImpl->setPlaceHolders(sampler, texture, bufferObject);
}

void VulkanDescriptorSetManager::unsetUniformBufferObject(uint32_t bindingIndex) {
    mImpl->unsetUniformBufferObject(bindingIndex);
}

void VulkanDescriptorSetManager::clearState() noexcept {
    mImpl->clearState();
}


}// namespace filament::backend
