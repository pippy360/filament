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

#ifndef TNT_FILAMENT_DESCRIPTORSETS_H
#define TNT_FILAMENT_DESCRIPTORSETS_H

#include <backend/DriverEnums.h>

#include <utils/CString.h>

#include <stdint.h>

namespace filament::descriptor_sets {

enum class DescriptorSet {
    PER_VIEW,
    PER_RENDERABLE,
    PER_MATERIAL,
};

uint8_t getIndex(DescriptorSet set) noexcept;

backend::DescriptorSetLayout const& getLayout(DescriptorSet set) noexcept;

utils::CString getDescriptorName(uint8_t set, uint8_t binding) noexcept;

} // namespace filament::descriptor_sets


#endif //TNT_FILAMENT_DESCRIPTORSETS_H
