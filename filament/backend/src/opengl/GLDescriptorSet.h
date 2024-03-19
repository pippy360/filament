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

#ifndef TNT_FILAMENT_BACKEND_OPENGL_GLDESCRIPTORSET_H
#define TNT_FILAMENT_BACKEND_OPENGL_GLDESCRIPTORSET_H

#include "DriverBase.h"

#include "gl_headers.h"

#include <backend/DriverEnums.h>

#include <utils/bitset.h>
#include <utils/FixedCapacityVector.h>

#include <variant>

#include <stddef.h>
#include <stdint.h>

namespace filament::backend {

struct GLBufferObject;
struct GLTexture;
struct GLDescriptorSetLayout;
class OpenGLProgram;
class OpenGLContext;

struct GLDescriptorSet : public HwDescriptorSet {

    using HwDescriptorSet::HwDescriptorSet;

    GLDescriptorSet(OpenGLContext& gl, GLDescriptorSetLayout const* layout) noexcept;

    // update a buffer descriptor in the set
    void update(OpenGLContext& gl,
            descriptor_binding_t binding, GLBufferObject* bo, size_t offset, size_t size) noexcept;

    // update a sampler descriptor in the set
    void update(OpenGLContext& gl,
            descriptor_binding_t binding, GLTexture* t, SamplerParams params) noexcept;

    // conceptually bind the set to the command buffer
    void bind(OpenGLContext& gl, OpenGLProgram* p,
            descriptor_set_t set, uint32_t const* offsets) const noexcept;

private:
    // a Buffer Descriptor such as SSBO or UBO with static offset
    struct Buffer {
        Buffer() = default;
        explicit Buffer(GLenum target) noexcept : target(target) { }
        GLenum target;                          // 4
        GLuint id = 0;                          // 4
        uint32_t offset = 0;                    // 4
        uint32_t size = 0;                      // 4
    };

    // a Buffer Descriptor such as SSBO or UBO with dynamic offset
    struct DynamicBuffer {
        DynamicBuffer() = default;
        explicit DynamicBuffer(GLenum target) noexcept : target(target) { }
        GLenum target;                          // 4
        GLuint id = 0;                          // 4
        uint32_t offset = 0;                    // 4
        uint32_t size = 0;                      // 4
    };

    // a UBO descriptor for ES2
    struct BufferGLES2 {
        BufferGLES2() = default;
        explicit BufferGLES2(bool dynamicOffset) noexcept : dynamicOffset(dynamicOffset) { }
        GLBufferObject const* bo = nullptr;     // 8
        uint32_t offset = 0;                    // 4
        bool dynamicOffset = false;             // 4
    };

    // A sampler descriptor
    struct Sampler {
        GLTexture* t;                           // 8
        GLuint sampler = 0;                     // 4
        float anisotropy = 1.0f;                // 4
    };

    // A sampler descriptor for ES2
    struct SamplerGLES2 {
        GLTexture* t = nullptr;                 // 8
        SamplerParams params{};                 // 4
        float anisotropy = 1.0f;                // 4
    };
    struct Descriptor {
        std::variant<
                Buffer,
                DynamicBuffer,
                BufferGLES2,
                Sampler,
                SamplerGLES2> desc;
    };
    utils::FixedCapacityVector<Descriptor> descriptors;
    utils::bitset64 dynamicBuffers;
    static_assert(sizeof(Descriptor) <= 32);
};

} // namespace filament::backend

#endif //TNT_FILAMENT_BACKEND_OPENGL_GLDESCRIPTORSET_H
