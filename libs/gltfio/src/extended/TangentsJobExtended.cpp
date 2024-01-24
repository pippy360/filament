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

#include "TangentsJobExtended.h"

#include "AssetLoaderExtended.h"
#include "../GltfEnums.h"
#include "../FFilamentAsset.h"
#include "../Utility.h"

#include <geometry/TangentSpaceMesh.h>
#include <utils/StructureOfArrays.h>

#include <memory>
#include <unordered_map>

using namespace filament::gltfio;
using namespace filament;
using namespace filament::math;

namespace {

constexpr uint8_t POSITIONS = 0;
constexpr uint8_t TANGENTS = 1;
constexpr uint8_t COLORS = 2;
constexpr uint8_t NORMALS = 3;
constexpr uint8_t UV0_a = 4; // _a is added to avoid naming conflict with MaterialProvider.h
constexpr uint8_t UV1_a = 5;
constexpr uint8_t WEIGHTS = 6;
constexpr uint8_t JOINTS = 7;
constexpr uint8_t INVALID = 0xFF;

using DataType = std::variant<float2*, float3*, float4*, ushort4*>;
using AttributeDataMap = std::unordered_map<uint8_t, DataType>;

inline uint8_t toCode(Attribute attr) {
    if (attr.type == cgltf_attribute_type_normal && attr.index == 0) {
        return NORMALS;
    }
    if (attr.type == cgltf_attribute_type_tangent && attr.index == 0) {
        return TANGENTS;
    }
    if (attr.type == cgltf_attribute_type_color && attr.index == 0) {
        return COLORS;
    }
    if (attr.type == cgltf_attribute_type_position && attr.index == 0) {
        return POSITIONS;
    }
    if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) {
        return UV0_a;
    }
    if (attr.type == cgltf_attribute_type_texcoord && attr.index == 1) {
        return UV1_a;
    }
    if (attr.type == cgltf_attribute_type_weights && attr.index == 0) {
        return WEIGHTS;
    }
    if (attr.type == cgltf_attribute_type_joints && attr.index == 0) {
        return JOINTS;
    }
    // Otherwise, this is not an attribute supported by Filament.
    return INVALID;
}

namespace data {

template<typename T>
T get(AttributeDataMap const& data, uint8_t attr) {
    auto iter = data.find(attr);
    if (iter != data.end()) {
        return std::get<T>(iter->second);
    }
    return nullptr;
}

template<typename T>
void allocate(AttributeDataMap& data, uint8_t attr, size_t count) {
    assert_invariant(data.find(attr) == data.end());
    data[attr] = (T) malloc(sizeof(std::remove_pointer_t<T>) * count);
}

template<typename T>
void free(AttributeDataMap& data, uint8_t attr) {
    if (data.find(attr) == data.end()) {
        return;
    }
    std::free(std::get<T>(data[attr]));
}

template<typename T>
void unpack(cgltf_accessor const* accessor, size_t const vertexCount, T out) {
    assert_invariant(accessor->count == vertexCount);
    if constexpr(std::is_same_v<T, float2*> ||
            std::is_same_v<T, float3*> ||
            std::is_same_v<T, float4*>) {
        assert_invariant(accessor->buffer_view);

        uint8_t const* data = nullptr;
        if (accessor->buffer_view->has_meshopt_compression) {
            data = (uint8_t const*) accessor->buffer_view->data + accessor->offset;
        } else {
            data = (uint8_t const*) accessor->buffer_view->buffer->data + utility::computeBindingOffset(accessor);
        }

        size_t const floatCount = (sizeof(std::remove_pointer_t<T>) / sizeof(float)) * vertexCount;
        size_t const byteCount = floatCount * sizeof(float);

        assert_invariant(accessor->count * cgltf_num_components(accessor->type) * sizeof(float)
                == byteCount);
        if (utility::requiresConversion(accessor)) {
            cgltf_accessor_unpack_floats(accessor, (float*) out, floatCount);
        } else {
            std::memcpy(out, data, byteCount);
        }
    } else if constexpr(std::is_same_v<T, ushort4*>) {
        // TODO, finish
        assert(false);
    }
}

template<typename T>
void unpack(cgltf_accessor const* accessor, AttributeDataMap& data, uint8_t attr, size_t const vertexCount) {
    assert_invariant(accessor->count == vertexCount);
    assert_invariant(data.find(attr) != data.end());
    unpack(accessor, vertexCount, data::get<T>(data, attr));
}

template<typename T>
void add(AttributeDataMap& data, uint8_t attr, size_t const vertexCount, float3* addition) {
    assert_invariant(data.find(attr) != data.end());

    T datav = std::get<T>(data[attr]);
    for (size_t i = 0; i < vertexCount; ++i) {
        if constexpr(std::is_same_v<T, float2*>) {
            datav[i] += addition[i].xy;
        } else if constexpr(std::is_same_v<T, float3*>) {
            datav[i] += addition[i];
        } else if constexpr(std::is_same_v<T, float4*>) {
            datav[i].xyz += addition[i];
        }
    }
}

} // namespace data

using POSITIONS_TYPE = float3*;
using TANGENTS_TYPE = float4*;
using COLORS_TYPE = float4*;
using NORMALS_TYPE = float3*;
using UV0_TYPE = float2*;
using UV1_TYPE = float2*;
using WEIGHTS_TYPE = float4*;
using JOINTS_TYPE = ushort4*;

void destroy(AttributeDataMap& data) {
    data::free<POSITIONS_TYPE>(data, POSITIONS);
    data::free<TANGENTS_TYPE>(data, TANGENTS);
    data::free<COLORS_TYPE>(data, COLORS);
    data::free<NORMALS_TYPE>(data, NORMALS);
    data::free<UV0_TYPE>(data, UV0_a);
    data::free<UV1_TYPE>(data, UV1_a);
    data::free<WEIGHTS_TYPE>(data, WEIGHTS);
    data::free<JOINTS_TYPE>(data, JOINTS);
}

} // anonymous namespace

namespace filament::gltfio {

// This procedure is designed to run in an isolated job.
void TangentsJobExtended::run(Params* params) {
    cgltf_primitive const& prim = *params->in.prim;
    int const morphTargetIndex = params->in.morphTargetIndex;
    bool const isMorphTarget = morphTargetIndex != kMorphTargetUnused;

    // Extract the vertex count from the first attribute. All attributes must have the same count.
    assert_invariant(prim.attributes_count > 0);
    auto const vertexCount = prim.attributes[0].data->count;
    assert_invariant(vertexCount > 0);

    std::unordered_map<uint8_t, cgltf_accessor const*> accessors;
    std::unordered_map<uint8_t, cgltf_accessor const*> morphAccessors;
    AttributeDataMap attributes;

    std::unique_ptr<uint3[]> unpackedTriangles;

    for (cgltf_size aindex = 0; aindex < prim.attributes_count; aindex++) {
        cgltf_attribute const& attr = prim.attributes[aindex];
        if (auto const attrCode = toCode({attr.type, attr.index}); attrCode != INVALID) {
            accessors[attrCode] = attr.data;
        }
    }

    std::vector<float3> morphDelta;
    if (isMorphTarget) {
        auto const& morphTarget = prim.targets[morphTargetIndex];
        for (cgltf_size aindex = 0; aindex < morphTarget.attributes_count; aindex++) {
            cgltf_attribute const& attr = morphTarget.attributes[aindex];
            if (auto const attrCode = toCode({attr.type, attr.index}); attrCode != INVALID) {
                assert_invariant(accessors.find(attrCode) != accessors.end() &&
                        "Morph target data has no corresponding base vertex data.");
                morphAccessors[attrCode] = attr.data;
            }
        }
        morphDelta.resize(vertexCount);
    }
    using AuxType = geometry::TangentSpaceMesh::AuxAttribute;
    geometry::TangentSpaceMesh::Builder tob;
    tob.vertexCount(vertexCount);

    for (auto [attr, accessor]: accessors) {
        switch(attr) {
            case POSITIONS:
                data::allocate<POSITIONS_TYPE>(attributes, attr, vertexCount);
                data::unpack<POSITIONS_TYPE>(accessor, attributes, attr, vertexCount);
                if (auto itr = morphAccessors.find(attr); itr != morphAccessors.end()) {
                    data::unpack<float3*>(itr->second, vertexCount, morphDelta.data());
                    data::add<POSITIONS_TYPE>(attributes, attr, vertexCount, morphDelta.data());

                    // We stash the positions as colors so that they can be retrieved without change
                    // after the TBN algo, which might have remeshed the input.
                    data::allocate<COLORS_TYPE>(attributes, COLORS, vertexCount);
                    float4* storage = data::get<COLORS_TYPE>(attributes, COLORS);
                    for (size_t i = 0; i < vertexCount; i++) {
                        storage[i] = float4 { morphDelta[i], 0.0 };
                    }
                    tob.aux(AuxType::COLORS, storage);
                }
                tob.positions(data::get<POSITIONS_TYPE>(attributes, attr));
                break;
            case TANGENTS:
                data::allocate<TANGENTS_TYPE>(attributes, attr, vertexCount);
                data::unpack<TANGENTS_TYPE>(accessor, attributes, attr, vertexCount);
                if (auto itr = morphAccessors.find(attr); itr != morphAccessors.end()) {
                    data::unpack<float3*>(itr->second, vertexCount, morphDelta.data());
                    data::add<TANGENTS_TYPE>(attributes, attr, vertexCount, morphDelta.data());
                }
                tob.tangents(data::get<TANGENTS_TYPE>(attributes, attr));
                break;
            case NORMALS:
                data::allocate<NORMALS_TYPE>(attributes, attr, vertexCount);
                data::unpack<NORMALS_TYPE>(accessor, attributes, attr, vertexCount);
                if (auto itr = morphAccessors.find(attr); itr != morphAccessors.end()) {
                    data::unpack<float3*>(itr->second, vertexCount, morphDelta.data());
                    data::add<NORMALS_TYPE>(attributes, attr, vertexCount, morphDelta.data());
                }
                tob.normals(data::get<NORMALS_TYPE>(attributes, attr));
                break;
            case COLORS:
                data::allocate<COLORS_TYPE>(attributes, attr, vertexCount);
                data::unpack<COLORS_TYPE>(accessor, attributes, attr, vertexCount);
                tob.aux(AuxType::COLORS, data::get<COLORS_TYPE>(attributes, attr));
                break;
            case UV0_a:
                data::allocate<UV0_TYPE>(attributes, attr, vertexCount);
                data::unpack<UV0_TYPE>(accessor, attributes, attr, vertexCount);
                tob.uvs(data::get<UV0_TYPE>(attributes, attr));
                break;
            case UV1_a:
                data::allocate<UV1_TYPE>(attributes, attr, vertexCount);
                data::unpack<UV1_TYPE>(accessor, attributes, attr, vertexCount);
                tob.aux(AuxType::UV1, data::get<UV1_TYPE>(attributes, attr));
                break;
            case WEIGHTS:
                data::allocate<WEIGHTS_TYPE>(attributes, attr, vertexCount);
                data::unpack<WEIGHTS_TYPE>(accessor, attributes, attr, vertexCount);
                tob.aux(AuxType::WEIGHTS, data::get<WEIGHTS_TYPE>(attributes, attr));
                break;
            case JOINTS:
                data::allocate<JOINTS_TYPE>(attributes, attr, vertexCount);
                data::unpack<JOINTS_TYPE>(accessor, attributes, attr, vertexCount);
                tob.aux(AuxType::JOINTS, data::get<JOINTS_TYPE>(attributes, attr));
                break;
            default:
                break;
        }
    }

    size_t const triangleCount = prim.indices ? (prim.indices->count / 3) : (vertexCount / 3);
    unpackedTriangles.reset(new uint3[triangleCount]);

    if (prim.indices) {
        for (size_t tri = 0, j = 0; tri < triangleCount; ++tri) {
            auto& triangle = unpackedTriangles[tri];
            triangle.x = cgltf_accessor_read_index(prim.indices, j++);
            triangle.y = cgltf_accessor_read_index(prim.indices, j++);
            triangle.z = cgltf_accessor_read_index(prim.indices, j++);
        }
    } else {
        for (size_t tri = 0, j = 0; tri < triangleCount; ++tri) {
            auto& triangle = unpackedTriangles[tri];
            triangle.x = j++;
            triangle.y = j++;
            triangle.z = j++;
        }
    }

    tob.triangleCount(triangleCount);
    tob.triangles(unpackedTriangles.get());
    auto const mesh = tob.build();

    auto& out = params->out;
    out.vertexCount = mesh->getVertexCount();

    auto const cleanup = [&]() {
        destroy(attributes);
        geometry::TangentSpaceMesh::destroy(mesh);
    };

    out.triangleCount = mesh->getTriangleCount();
    out.triangles = (uint3*) malloc(out.triangleCount * sizeof(uint3));
    mesh->getTriangles(out.triangles);

    out.tbn = (short4*) malloc(out.vertexCount * sizeof(short4));
    mesh->getQuats(out.tbn);
    if (isMorphTarget) {
        // For morph targets, we need to retrieve the positions, but note that the unadjusted
        // positions are stored as colors.
        attributes = {{COLORS, (COLORS_TYPE) nullptr}};
    }

    for (auto [attr, data]: attributes) {
        switch (attr) {
            case POSITIONS:
                out.positions = (POSITIONS_TYPE) malloc(
                        out.vertexCount * sizeof(std::remove_pointer_t<POSITIONS_TYPE>));
                mesh->getPositions(out.positions);
                break;
            case COLORS: {
                if (!isMorphTarget) {
                    out.colors = (COLORS_TYPE) malloc(
                            out.vertexCount * sizeof(std::remove_pointer_t<COLORS_TYPE>));
                    mesh->getAux(AuxType::COLORS, out.colors);
                } else {
                    // For morph targets, we use COLORS as a way to store the original positions.
                    out.positions = (POSITIONS_TYPE) malloc(
                            out.vertexCount * sizeof(std::remove_pointer_t<POSITIONS_TYPE>));

                    std::vector<float4> scratch;
                    scratch.resize(out.vertexCount);
                    mesh->getAux(AuxType::COLORS, scratch.data());

                    for (size_t i = 0; i < out.vertexCount; ++i) {
                        out.positions[i] = scratch[i].xyz;
                    }
                }
                break;
            }
            case UV0_a:
                out.uv0 = (UV0_TYPE) malloc(
                        out.vertexCount * sizeof(std::remove_pointer_t<UV0_TYPE>));
                mesh->getUVs(out.uv0);
                break;
            case UV1_a:
                out.uv1 = (UV1_TYPE) malloc(
                        out.vertexCount * sizeof(std::remove_pointer_t<UV1_TYPE>));
                mesh->getAux(AuxType::UV1, out.uv1);
                break;
            case WEIGHTS:
                out.weights = (WEIGHTS_TYPE) malloc(
                        out.vertexCount * sizeof(std::remove_pointer_t<WEIGHTS_TYPE>));
                mesh->getAux(AuxType::WEIGHTS, out.weights);
                break;
            case JOINTS:
                out.joints = (JOINTS_TYPE) malloc(
                        out.vertexCount * sizeof(std::remove_pointer_t<JOINTS_TYPE>));
                mesh->getAux(AuxType::JOINTS, out.joints);
                break;
            default:
                break;
        }
    }
    cleanup();
}

} // namespace filament::gltfio
