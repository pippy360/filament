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

#ifndef GLTFIO_ASSETLOADEREXTENDED_H
#define GLTFIO_ASSETLOADEREXTENDED_H

#include "../FFilamentAsset.h"

#include <backend/BufferDescriptor.h>
#include <gltfio/AssetLoader.h>

#include <cgltf.h>

#include <future>
#include <string>


namespace filament::gltfio {

struct Primitive;
struct FFilamentAsset;
class DracoCache;

using BufferDescriptor = filament::backend::BufferDescriptor;

// The cgltf attribute is a type and the attribute index
struct Attribute {
    cgltf_attribute_type type;
    int index;
};

// The Filament Attribute is defined as a type, a slot, and whether the attribute is normalized or not
struct FilamentAttribute {
    VertexAttribute attribute;
    int slot;
    bool normalized;
};

struct AttributeHash {
    size_t operator()(Attribute const& key) const {
        size_t h1 = std::hash<uint64_t>{}((uint64_t) key.type);
        size_t h2 = std::hash<uint64_t>{}((uint64_t) key.index);
        return h1 ^ (h2 << 1);
    }
};

struct AttributeEqual {
    bool operator()(Attribute const& lhs, Attribute const& rhs) const {
        return lhs.type == rhs.type && lhs.index == rhs.index;
    }
};

using AttributesMap =
        std::unordered_map<Attribute, FilamentAttribute, AttributeHash, AttributeEqual>;

struct AssetLoaderExtended {

    using BufferSlot = FFilamentAsset::ResourceInfoExtended::BufferSlot;    
    using Output = Primitive;

    struct Input {
        cgltf_data* gltf;
        cgltf_primitive* prim;
        char const* name;
        DracoCache* dracoCache;
        Material* material;
    };

    AssetLoaderExtended(AssetConfigurationExtended const& config, Engine* engine,
            MaterialProvider& materials)
        : mEngine(engine), mGltfPath(config.gltfPath), mMaterials(materials) {}

    ~AssetLoaderExtended() = default;

    bool createPrimitive(Input* input, Output* out, std::vector<BufferSlot>& outSlots);
    
private:
    filament::Engine* mEngine;
    std::string mGltfPath;
    MaterialProvider& mMaterials;
};

} // namespace filament::gltfio

#endif // GLTFIO_ASSETLOADEREXTENDED_H
