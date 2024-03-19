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

#ifndef TNT_FILAMENT_DESCRIPTORSETLAYOUT_H
#define TNT_FILAMENT_DESCRIPTORSETLAYOUT_H

#include <backend/DriverEnums.h>

#include <backend/DriverApiForward.h>
#include <backend/Handle.h>

#include "details/Engine.h"

#include <utils/BitmaskEnum.h>

#include <utility>

#include <stddef.h>
#include <stdint.h>

namespace filament {

class DescriptorSetLayout {
public:
    DescriptorSetLayout(backend::DriverApi& driver,
            backend::DescriptorSetLayout descriptorSetLayout) noexcept {
        mDescriptorCount = descriptorSetLayout.bindings.size();
        for (auto&& desc : descriptorSetLayout.bindings) {
            if (any(desc.flags & backend::DescriptorFlags::DYNAMIC_OFFSET)) {
                mDynamicBuffers.set(desc.binding);
                mDynamicBufferCount++;
            }
            if (desc.type == backend::DescriptorType::SAMPLER) {
                mSamplerCount++;
            } else {
                mBufferCount++;
            }
        }

        mDescriptorSetLayoutHandle = driver.createDescriptorSetLayout(
                std::move(descriptorSetLayout));
    }

    backend::DescriptorSetLayoutHandle getHandle() const noexcept {
        return mDescriptorSetLayoutHandle;
    }

    size_t getDynamicBufferCount() const noexcept {
        return mDynamicBufferCount;
    }

    size_t getBufferCount() const noexcept {
        return mBufferCount;
    }

    size_t getSamplerCount() const noexcept {
        return mSamplerCount;
    }

    size_t getDescriptorCount() const noexcept {
        return mDescriptorCount;
    }

    utils::bitset64 getDynamicBuffers() const noexcept {
        return mDynamicBuffers;
    }

private:
    backend::DescriptorSetLayoutHandle mDescriptorSetLayoutHandle;
    utils::bitset64 mDynamicBuffers;
    uint8_t mDynamicBufferCount;
    uint8_t mBufferCount;
    uint8_t mSamplerCount;
    uint8_t mDescriptorCount;
};


} // namespace filament

#endif //TNT_FILAMENT_DESCRIPTORSETLAYOUT_H
