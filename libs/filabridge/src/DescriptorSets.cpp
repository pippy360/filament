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

#include "private/filament/DescriptorSets.h"

#include <backend/DriverEnums.h>

#include <utils/CString.h>
#include <utils/debug.h>

#include <string_view>

#include <stdint.h>

namespace filament::descriptor_sets {

using namespace backend;

static backend::DescriptorSetLayout perViewDescriptorSetLayout = {{
    { DescriptorType::UNIFORM_BUFFER, ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  0, DescriptorFlags::NONE, 0 },
    { DescriptorType::UNIFORM_BUFFER, ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  1, DescriptorFlags::NONE, 0 },
    { DescriptorType::UNIFORM_BUFFER, ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  2, DescriptorFlags::NONE, 0 },
    { DescriptorType::UNIFORM_BUFFER, ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  3, DescriptorFlags::NONE, 0 },
    { DescriptorType::UNIFORM_BUFFER, ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  4, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,                                   ShaderStageFlags::FRAGMENT,  5, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,                                   ShaderStageFlags::FRAGMENT,  6, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,                                   ShaderStageFlags::FRAGMENT,  7, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,                                   ShaderStageFlags::FRAGMENT,  8, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,                                   ShaderStageFlags::FRAGMENT,  9, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,                                   ShaderStageFlags::FRAGMENT, 10, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,                                   ShaderStageFlags::FRAGMENT, 11, DescriptorFlags::NONE, 0 },
}};

static backend::DescriptorSetLayout perRenderableDescriptorSetLayout = {{
    { DescriptorType::UNIFORM_BUFFER, ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  0, DescriptorFlags::DYNAMIC_OFFSET, 0 },
    { DescriptorType::UNIFORM_BUFFER, ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  1, DescriptorFlags::DYNAMIC_OFFSET, 0 },
    { DescriptorType::UNIFORM_BUFFER, ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  2, DescriptorFlags::NONE,           0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX                             ,  3, DescriptorFlags::NONE,           0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX                             ,  4, DescriptorFlags::NONE,           0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX                             ,  5, DescriptorFlags::NONE,           0 },
}};

// FIXME: more samplers at higher feature levels
//  maximum of 12 sampler when unlit at feature level 1
static backend::DescriptorSetLayout perMaterialDescriptorSetLayout = {{
    { DescriptorType::UNIFORM_BUFFER, ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  0, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  1, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  2, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  3, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  4, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  5, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  6, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  7, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  8, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,  9, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT, 10, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT, 11, DescriptorFlags::NONE, 0 },
    { DescriptorType::SAMPLER,        ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT, 12, DescriptorFlags::NONE, 0 },
}};

backend::DescriptorSetLayout const& getLayout(DescriptorSet set) noexcept {
    switch (set) {
        case DescriptorSet::PER_VIEW:
            return perViewDescriptorSetLayout;
        case DescriptorSet::PER_RENDERABLE:
            return perRenderableDescriptorSetLayout;
        case DescriptorSet::PER_MATERIAL:
            return perMaterialDescriptorSetLayout;
    }
}

uint8_t getIndex(DescriptorSet set) noexcept {
    switch (set) {
        case DescriptorSet::PER_VIEW:
            return 0;
        case DescriptorSet::PER_RENDERABLE:
            return 1;
        case DescriptorSet::PER_MATERIAL:
            return 2;
    }
}

utils::CString getDescriptorName(uint8_t set, uint8_t binding) noexcept {
    using namespace std::literals;
    constexpr const std::string_view set0[] = {
            "FrameUniforms"sv,
            "LightsUniforms"sv,
            "ShadowUniforms"sv,
            "FroxelRecordUniforms"sv,
            "FroxelsUniforms"sv,
            "light_shadowMap"sv,
            "light_iblDFG"sv,
            "light_iblSpecular"sv,
            "light_ssao"sv,
            "light_ssr"sv,
            "light_structure"sv,
            "light_fog"sv,
    };
    constexpr const std::string_view set1[] = {
            "ObjectUniforms"sv,
            "BonesUniforms"sv,
            "MorphingUniforms"sv,
            "morphTargetBuffer_positions"sv,
            "morphTargetBuffer_tangents"sv,
            "monesBuffer_indicesAndWeights"sv,
    };
    if (set == 0) {
        assert_invariant(binding < perViewDescriptorSetLayout.bindings.size());
        std::string_view const& s = set0[binding];
        return { s.data(), s.size() };
    }
    if (set == 1) {
        assert_invariant(binding < perRenderableDescriptorSetLayout.bindings.size());
        std::string_view const& s = set1[binding];
        return { s.data(), s.size() };
    }
    if (set == 3) {
        assert_invariant(binding < 1);
        return "MaterialParams";
    }
    return {};
}

} // namespace filament::descriptor_sets
