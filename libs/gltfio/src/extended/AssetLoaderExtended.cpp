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

#include "AssetLoaderExtended.h"

#include "TangentsJobExtended.h"
#include "../DracoCache.h"
#include "../FFilamentAsset.h"
#include "../GltfEnums.h"
#include "../Utility.h"

#include <filament/BufferObject.h>

#include <utils/JobSystem.h>
#include <utils/Log.h>
#include <utils/Panic.h>

#include <cgltf.h>

#include <future>

namespace filament::gltfio {

namespace {

template<typename BufferType>
struct BufferProducer {
    void setValue(BufferType buffer) {
        mPromise.set_value(buffer);
    }
    std::future<BufferType> getFuture() { return std::move(mPromise.get_future()); }

protected:
    static_assert(MAX_MORPH_TARGETS <= 256);
    std::promise<BufferType> mPromise;
};


constexpr uint8_t const VERTEX_JOB = 0x1;
constexpr uint8_t const INDEX_JOB = 0x2;
constexpr uint8_t const MORPH_TARGET_JOB = 0x4;

constexpr int const GENERATED_0 = FFilamentAsset::ResourceInfoExtended::GENERATED_0_INDEX;
constexpr int const GENERATED_1 = FFilamentAsset::ResourceInfoExtended::GENERATED_1_INDEX;

using BufferSlot = AssetLoaderExtended::BufferSlot;

struct PrimitiveWorkload {
    PrimitiveWorkload(cgltf_primitive const* primitive) : primitive(primitive) {}

    cgltf_primitive const* primitive;

    uint8_t jobs = 0;
    AttributesMap attributesMap;

};

using BufferType = std::variant<short4*, ushort4*, float2*, float3*, float4*>;

inline void* getRawPointer(BufferType buffer) {
    if (std::holds_alternative<short4*>(buffer)) {
        return std::get<short4*>(buffer);
    } else if (std::holds_alternative<ushort4*>(buffer)) {
        return std::get<ushort4*>(buffer);
    } else if (std::holds_alternative<float2*>(buffer)) {
        return std::get<float2*>(buffer);
    } else if (std::holds_alternative<float3*>(buffer)) {
        return std::get<float3*>(buffer);
    } else if (std::holds_alternative<float4*>(buffer)) {
        return std::get<float4*>(buffer);
    }
    return nullptr;
 }

inline size_t getSize(BufferType buffer)  {
    if (std::holds_alternative<short4*>(buffer)) {
        return sizeof(short4);
    } else if (std::holds_alternative<ushort4*>(buffer)) {
        return sizeof(ushort4);
    } else if (std::holds_alternative<float2*>(buffer)) {
        return sizeof(float2);
    } else if (std::holds_alternative<float3*>(buffer)) {
        return sizeof(float3);
    } else if (std::holds_alternative<float4*>(buffer)) {
        return sizeof(float4);
    }
    return 0;
}

inline VertexBuffer::AttributeType getType(BufferType buffer) {
    if (std::holds_alternative<short4*>(buffer)) {
        return VertexBuffer::AttributeType::SHORT4;
    } else if (std::holds_alternative<ushort4*>(buffer)) {
        return VertexBuffer::AttributeType::USHORT4;
    } else if (std::holds_alternative<float2*>(buffer)) {
        return VertexBuffer::AttributeType::FLOAT2;
    } else if (std::holds_alternative<float3*>(buffer)) {
        return VertexBuffer::AttributeType::FLOAT3;
    } else if (std::holds_alternative<float4*>(buffer)) {
        return VertexBuffer::AttributeType::FLOAT4;
    }
    return VertexBuffer::AttributeType::FLOAT4;
    //utils::PANIC_POSTCONDITION("Unexpected buffer type");
}

inline std::tuple<VertexBuffer::AttributeType, size_t, void*> getVertexBundle(
        VertexAttribute attrib, TangentsJobExtended::OutputParams const& out) {
    VertexBuffer::AttributeType type;
    size_t byteCount = 0;
    void* data = nullptr;
    switch (attrib) {
        case VertexAttribute::POSITION:
            type = getType(out.positions);
            byteCount = getSize(out.positions);
            data = getRawPointer(out.positions);
            break;
        case VertexAttribute::TANGENTS:
            type = getType(out.tbn);
            byteCount = getSize(out.tbn);
            data = getRawPointer(out.tbn);                        
            break;
        case VertexAttribute::COLOR:
            type = getType(out.colors);
            byteCount = getSize(out.colors);
            data = getRawPointer(out.colors);                                                
            break;
        case VertexAttribute::UV0:
            type = getType(out.uv0);
            byteCount = getSize(out.uv0);
            data = getRawPointer(out.uv0);
            break;
        case VertexAttribute::UV1:
            type = getType(out.uv1);
            byteCount = getSize(out.uv1);
            data = getRawPointer(out.uv1);
            break;
        case VertexAttribute::BONE_INDICES:
            type = getType(out.joints);
            byteCount = getSize(out.joints);
            data = getRawPointer(out.joints);                        
            break;
        case VertexAttribute::BONE_WEIGHTS:
            type = getType(out.weights);
            byteCount = getSize(out.weights);
            data = getRawPointer(out.weights);                        
            break;
        default:
            PANIC_POSTCONDITION("Unexpected vertex attribute %d",
                    static_cast<int>(attrib));
    }
    return {type, byteCount, data};
}

// This will run the jobs to create tangent spaces if necessary, or simply forward the data if the
// input does not require processing. The output is a list of buffers that will be uploaded in the
// ResourceLoader.
std::vector<BufferSlot> computeTangents(cgltf_primitive const* prim, uint8_t const jobType,
        AttributesMap const& attributesMap,
        // A set of morph targets to generate tangents for.
        std::vector<int> const& morphTargets, filament::Engine* engine) {
    using Params = TangentsJobExtended::Params;

    std::unordered_map<int, Params> jobs;
    auto getJob = [&jobs](int key) -> Params& {
        auto result = jobs.find(key);
        if (result == jobs.end()) {
            return (jobs.emplace(key, Params{}).first)->second;
        }
        return result->second;
    };

    utils::slog.e <<"jobs=" << static_cast<int>(jobType) << utils::io::endl;
    
    // Create a job description for each triangle-based primitive.
    // Collect all TANGENT vertex attribute slots that need to be populated.
    if ((jobType & VERTEX_JOB) != 0) {
        auto& job = getJob(TangentsJobExtended::kMorphTargetUnused);
        job.in = { prim, attributesMap };
        job.jobType |= VERTEX_JOB;
    }
    if ((jobType & INDEX_JOB) != 0) {
        auto& job = getJob(TangentsJobExtended::kMorphTargetUnused);
        job.in = { prim, attributesMap };
        job.jobType |= INDEX_JOB;
    }
    for (auto const target : morphTargets) {
        auto& job = getJob(target);
        job.jobType = MORPH_TARGET_JOB,        
        job.in = {
                .prim = prim,
                .morphTargetIndex = target,
        };
    }

    utils::slog.e <<"n jobs=" << jobs.size() << utils::io::endl;

    utils::JobSystem& js = engine->getJobSystem();
    utils::JobSystem::Job* parent = js.createJob();
    for (auto& [key, params]: jobs) {
        js.run(utils::jobs::createJob(js, parent,
                [pptr = &params] { TangentsJobExtended::run(pptr); }));
    }
    js.runAndWait(parent);

    std::vector<BufferSlot> slots;

    struct MorphTargetOut{
        int morphTarget;
        float3* positions;
        short4* tbn;
        size_t vertexCount;
    };
    std::vector<MorphTargetOut> morphTargetOuts;

    utils::slog.e <<"a slot=" << slots.size() << utils::io::endl;

    for (auto& [key, params]: jobs) {
        uint8_t const jobType = params.jobType;
        TangentsJobExtended::OutputParams const& out = params.out;
        size_t const vertexCount = out.vertexCount;

        if ((jobType & VERTEX_JOB) != 0) {
            utils::slog.e <<"there vertices" << utils::io::endl;
            auto vertexBufferBuilder = VertexBuffer::Builder().enableBufferObjects().vertexCount(
                    vertexCount);

            std::vector<BufferSlot> vslots;
            for (auto [cgltfAttr, filamentAttr]: params.in.attributesMap) {
                auto const [vattr, slot, normalize] = filamentAttr;
                auto const [type, byteCount, data] = getVertexBundle(vattr, out);
                vertexBufferBuilder.attribute(vattr, slot, type);
                if (normalize) {
                    vertexBufferBuilder.normalized(vattr);
                }
                utils::slog.e << "attr: " << static_cast<int>(vattr) << " "
                              << (int) slot << " " <<  (int) type << " "
                              << (int) byteCount << " " << data << utils::io::endl;
                vslots.push_back({
                    .slot = slot,
                    .sizeInBytes = byteCount * vertexCount,
                    .data = data,
                });
            }

            assert_invariant(!vslots.empty());
            vertexBufferBuilder.bufferCount(vslots.size());
            auto vertexBuffer = vertexBufferBuilder.build(*engine);
            std::for_each(vslots.begin(), vslots.end(), [vertexBuffer](BufferSlot& slot) {
                slot.vertices = vertexBuffer;
            });
            slots.insert(slots.end(), vslots.begin(), vslots.end());
        }
        if ((jobType & INDEX_JOB) != 0) {
            utils::slog.e <<"there indices" << utils::io::endl;
            
            auto indexBuffer = IndexBuffer::Builder()
                    .indexCount(out.triangleCount * 3)
                    .bufferType(IndexBuffer::IndexType::UINT)
                    .build(*engine);

            slots.push_back({
                .indices = indexBuffer,
                .sizeInBytes = out.triangleCount * 3 * 4,
                .data = out.triangles,
            });
        }
        if ((jobType & MORPH_TARGET_JOB) != 0) {
            morphTargetOuts.push_back({
                .morphTarget = params.in.morphTargetIndex,
                .positions = out.positions,
                .tbn = out.tbn,
                .vertexCount = vertexCount,
            });
        }
    }
    
    if (!morphTargets.empty()) {
        auto const vertexCount = morphTargetOuts[0].vertexCount;
        MorphTargetBuffer* buffer = MorphTargetBuffer::Builder()
                .count(morphTargets.size())
                .vertexCount(vertexCount)
                .build(*engine);
        for (auto target : morphTargetOuts) {
            assert_invariant(target.vertexCount == vertexCount);
            slots.push_back({
                .target = buffer,
                .slot = target.morphTarget,
                .targetData = {
                        .tbn = target.tbn,
                        .positions = target.positions,
                }
            });
        }
    }
    return slots;
}

} // anonymous namespace

bool AssetLoaderExtended::createPrimitive(Input* input, Output* out, std::vector<BufferSlot>& outSlots) {    
    auto gltf = input->gltf;
    auto prim = input->prim;
    auto name = input->name;

    uint8_t jobType = 0;

    // In glTF, each primitive may or may not have an index buffer.
    const cgltf_accessor* accessor = prim->indices;
    if (accessor || prim->attributes_count > 0) {
        IndexBuffer::IndexType indexType;
        if (!getIndexType(accessor->component_type, &indexType)) {
            utils::slog.e << "Unrecognized index type in " << name << utils::io::endl;
            return false;
        }
        jobType |= INDEX_JOB;
    }

    jobType |= VERTEX_JOB;    

    AttributesMap attributesMap;
    bool hasUv0 = false, hasUv1 = false, hasVertexColor = false, hasNormals = false;
    int slotCount = 0;

    for (cgltf_size aindex = 0; aindex < prim->attributes_count; aindex++) {
        cgltf_attribute const attribute = prim->attributes[aindex];
        int const index = attribute.index;
        cgltf_attribute_type const atype = attribute.type;
        cgltf_accessor const* accessor = attribute.data;

        Attribute const cattr { atype, index };

        // At a minimum, surface orientation requires normals to be present in the source data.
        // Here we re-purpose the normals slot to point to the quats that get computed later.
        if (atype == cgltf_attribute_type_normal) {
            if (!hasNormals) {
                FilamentAttribute const fattr{VertexAttribute::TANGENTS, slotCount++,
                        true};
                hasNormals = true;
                attributesMap[cattr] = fattr;
            }
            continue;
        }

        if (atype == cgltf_attribute_type_tangent) {
            if (!hasNormals) {
                FilamentAttribute const fattr{VertexAttribute::TANGENTS, slotCount++,
                        true};
                hasNormals = true;
                attributesMap[cattr] = fattr;
            }
            continue;
        }

        // Translate the cgltf attribute enum into a Filament enum.
        VertexAttribute semantic;
        if (!getVertexAttrType(atype, &semantic)) {
            utils::slog.e << "Unrecognized vertex semantic in " << name << utils::io::endl;
            return false;
        }
        if (atype == cgltf_attribute_type_weights && index > 0) {
            utils::slog.e << "Too many bone weights in " << name << utils::io::endl;
            continue;
        }
        if (atype == cgltf_attribute_type_joints && index > 0) {
            utils::slog.e << "Too many joints in " << name << utils::io::endl;
            continue;
        }
        if (atype == cgltf_attribute_type_texcoord) {
            if (index >= UvMapSize) {
                utils::slog.e << "Too many texture coordinate sets in " << name << utils::io::endl;
                continue;
            }
            UvSet uvset = out->uvmap[index];
            switch (uvset) {
                case UV0:
                    semantic = VertexAttribute::UV0;
                    hasUv0 = true;
                    break;
                case UV1:
                    semantic = VertexAttribute::UV1;
                    hasUv1 = true;
                    break;
                case UNUSED:
                    // If we have a free slot, then include this unused UV set in the VertexBuffer.
                    // This allows clients to swap the glTF material with a custom material.
                    if (!hasUv0 && getNumUvSets(out->uvmap) == 0) {
                        semantic = VertexAttribute::UV0;
                        hasUv0 = true;
                        break;
                    }

                    // If there are no free slots then drop this unused texture coordinate set.
                    // This should not print an error or warning because the glTF spec stipulates an
                    // order of degradation for gracefully dropping UV sets. We implement this in
                    // constrainMaterial in MaterialProvider.
                    continue;
            }
        }

        if (atype == cgltf_attribute_type_color) {
            hasVertexColor = true;
        }

        // The positions accessor is required to have min/max properties, use them to expand
        // the bounding box for this primitive.
        if (atype == cgltf_attribute_type_position) {
            const float* minp = &accessor->min[0];
            const float* maxp = &accessor->max[0];
            out->aabb.min = min(out->aabb.min, float3(minp[0], minp[1], minp[2]));
            out->aabb.max = max(out->aabb.max, float3(maxp[0], maxp[1], maxp[2]));
        }

        if (VertexBuffer::AttributeType fatype, actualType;
                !getElementType(accessor->type, accessor->component_type, &fatype, &actualType)) {
            utils::slog.e << "Unsupported accessor type in " << name << utils::io::endl;
            return false;
        }

        // The cgltf library provides a stride value for all accessors, even though they do not
        // exist in the glTF file. It is computed from the type and the stride of the buffer view.
        // As a convenience, cgltf also replaces zero (default) stride with the actual stride.
        // const int stride = (fatype == actualType) ? accessor->stride : 0;
        attributesMap[cattr] = { semantic, slotCount++ };
    }

    cgltf_size targetsCount = prim->targets_count;
    if (targetsCount > MAX_MORPH_TARGETS) {
        utils::slog.w << "WARNING: Exceeded max morph target count of "
                << MAX_MORPH_TARGETS << utils::io::endl;
        targetsCount = MAX_MORPH_TARGETS;
    }

    // A set of morph targets to generate tangents for.
    std::vector<int> morphTargets;
    
    Aabb const baseAabb(out->aabb);
    for (cgltf_size targetIndex = 0; targetIndex < targetsCount; targetIndex++) {
        bool morphTargetHasNormals = false;
        cgltf_morph_target const& target = prim->targets[targetIndex];
        for (cgltf_size aindex = 0; aindex < target.attributes_count; aindex++) {
            cgltf_attribute const& attribute = target.attributes[aindex];
            cgltf_accessor const* accessor = attribute.data;
            cgltf_attribute_type const atype = attribute.type;

            if (atype != cgltf_attribute_type_position && atype != cgltf_attribute_type_normal &&
                    atype != cgltf_attribute_type_tangent) {
                utils::slog.e << "Only positions, normals, and tangents can be morphed."
                              << " type=" << static_cast<int>(atype) << utils::io::endl;
                return false;
            }

            if (VertexBuffer::AttributeType fatype, actualType; !getElementType(accessor->type,
                        accessor->component_type, &fatype, &actualType)) {
                utils::slog.e << "Unsupported accessor type in " << name << utils::io::endl;
                return false;
            }

            if (atype == cgltf_attribute_type_position && accessor->has_min && accessor->has_max) {
                Aabb targetAabb(baseAabb);
                float const* minp = &accessor->min[0];
                float const* maxp = &accessor->max[0];

                // We assume that the range of morph target weight is [0, 1].
                targetAabb.min += float3(minp[0], minp[1], minp[2]);
                targetAabb.max += float3(maxp[0], maxp[1], maxp[2]);

                out->aabb.min = min(out->aabb.min, targetAabb.min);
                out->aabb.max = max(out->aabb.max, targetAabb.max);
            }

            if (atype == cgltf_attribute_type_tangent) {
                morphTargetHasNormals = true;
                morphTargets.push_back(targetIndex);
            }
        }
        // Generate flat normals if necessary.
        if (!morphTargetHasNormals && prim->material && !prim->material->unlit) {
            morphTargets.push_back(targetIndex);
        }
    }

    auto vertexCount = accessor->count;
    if (vertexCount == 0) {
        utils::slog.e << "Empty vertex buffer in " << name << utils::io::endl;
        return false;
    }

    // We provide a single dummy buffer (filled with 0xff) for all unfulfilled vertex requirements.
    // The color data should be a sequence of normalized UBYTE4, so dummy UVs are USHORT2 to make
    // the sizes match.

    if (mMaterials.needsDummyData(VertexAttribute::UV0) && !hasUv0) {
        attributesMap[{cgltf_attribute_type_texcoord, GENERATED_0}] = {VertexAttribute::UV0,
                slotCount, true};
    }

    if (mMaterials.needsDummyData(VertexAttribute::UV1) && !hasUv1) {
        attributesMap[{cgltf_attribute_type_texcoord, GENERATED_1}] = {VertexAttribute::UV1,
                slotCount, true};
    }

    if (mMaterials.needsDummyData(VertexAttribute::COLOR) && !hasVertexColor) {
        attributesMap[{cgltf_attribute_type_color, GENERATED_0}] = {VertexAttribute::COLOR,
                slotCount, true};
    }

    int numUvSets = getNumUvSets(out->uvmap);
    if (!hasUv0 && numUvSets > 0) {
        attributesMap[{cgltf_attribute_type_texcoord, GENERATED_0}] = {
                VertexAttribute::UV0, slotCount, true};
    }

    if (!hasUv1 && numUvSets > 1) {
        utils::slog.w << "Missing UV1 data in " << name << utils::io::endl;
        attributesMap[{cgltf_attribute_type_texcoord, GENERATED_1}] = {
                VertexAttribute::UV1, slotCount, true};
    }

    if (!utility::loadCgltfBuffers(gltf, mGltfPath.c_str(), nullptr)) {
        return false;
    }

    utility::decodeDracoMeshes(gltf, prim, input->dracoCache);
    utility::decodeMeshoptCompression(gltf);

    auto& slots = outSlots = computeTangents(prim, jobType, attributesMap, morphTargets, mEngine);

    for (auto slot: slots) {
        if (slot.vertices) {
            assert_invariant(!out->vertices || out->vertices == slot.vertices);
            out->vertices = slot.vertices;
        }
        if (slot.indices) {
            assert_invariant(!out->indices || out->indices == slot.indices);
            out->indices = slot.indices;
        }
        if (slot.target) {
            assert_invariant(!out->targets || out->targets == slot.target);
            out->targets = slot.target;
        }
    }
    return true;
}

} // namespace filament::gltfio
