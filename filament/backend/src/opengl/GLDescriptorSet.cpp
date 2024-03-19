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

#include "GLDescriptorSet.h"

#include "GLBufferObject.h"
#include "GLDescriptorSetLayout.h"
#include "GLTexture.h"
#include "GLUtils.h"
#include "OpenGLContext.h"
#include "OpenGLProgram.h"

#include "gl_headers.h"

#include <backend/DriverEnums.h>

#include <utils/bitset.h>
#include <utils/compiler.h>
#include <utils/debug.h>
#include <utils/Log.h>
#include <utils/BitmaskEnum.h>

#include <algorithm>
#include <type_traits>
#include <variant>

#include <stddef.h>
#include <stdint.h>

namespace filament::backend {

GLDescriptorSet::GLDescriptorSet(OpenGLContext& gl,
        GLDescriptorSetLayout const* layout) noexcept
        : descriptors(layout->bindings.size()) {

    // We have allocated enough storage for all descriptors. Now allocate the empty descriptor
    // themselves.
    for (auto const& entry : layout->bindings) {
        size_t const index = entry.binding;

        // now we'll initialize the alternative for each way we can handle this descriptor.
        auto& desc = descriptors.back().desc;
        switch (entry.type) {
            case DescriptorType::UNIFORM_BUFFER: {
                // A uniform buffer can have dynamic offsets or not and have special handling for
                // ES2 (where we need to emulate it). That's four alternatives.
                bool const dynamicOffset = any(entry.flags & DescriptorFlags::DYNAMIC_OFFSET);
                dynamicBuffers.set(index, dynamicOffset);
                if (UTILS_UNLIKELY(gl.isES2())) {
                    desc.emplace<BufferGLES2>(dynamicOffset);
                } else {
                    auto const type = GLUtils::getBufferBindingType(BufferObjectBinding::UNIFORM);
                    if (dynamicOffset) {
                        desc.emplace<DynamicBuffer>(type);
                    } else {
                        desc.emplace<Buffer>(type);
                    }
                }
                break;
            }
            case DescriptorType::SHADER_STORAGE_BUFFER: {
                // shader storage buffers are not supported on ES2, So that's two alternatives.
                auto const type = GLUtils::getBufferBindingType(
                        BufferObjectBinding::SHADER_STORAGE);
                if (any(entry.flags & DescriptorFlags::DYNAMIC_OFFSET)) {
                    desc.emplace<DynamicBuffer>(type);
                } else {
                    desc.emplace<Buffer>(type);
                }
                break;
            }
            case DescriptorType::SAMPLER:
                if (UTILS_UNLIKELY(gl.isES2())) {
                    desc.emplace<SamplerGLES2>();
                } else {
                    desc.emplace<Sampler>();
                }
                break;
        }
    }
}

void GLDescriptorSet::update(OpenGLContext&,
        descriptor_binding_t binding, GLBufferObject* bo, size_t offset, size_t size) noexcept {

    assert_invariant(binding < descriptors.size());
    std::visit([=](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, Buffer> || std::is_same_v<T, DynamicBuffer>) {
            arg.id = bo->gl.id;
            arg.offset = uint32_t(offset);
            arg.size = uint32_t(size);
        } else if constexpr (std::is_same_v<T, BufferGLES2>) {
            arg.bo = bo;
            arg.offset = uint32_t(offset);
        }
    }, descriptors[binding].desc);
}

void GLDescriptorSet::update(OpenGLContext& gl,
        descriptor_binding_t binding, GLTexture* t, SamplerParams params) noexcept {

    assert_invariant(binding < descriptors.size());
    std::visit([=, &gl](auto&& arg) mutable {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, Sampler> || std::is_same_v<T, SamplerGLES2>) {
            if (UTILS_UNLIKELY(t->target == SamplerType::SAMPLER_EXTERNAL)) {
                // From OES_EGL_image_external spec:
                // "The default s and t wrap modes are CLAMP_TO_EDGE, and it is an INVALID_ENUM
                //  error to set the wrap mode to any other value."
                params.wrapS = SamplerWrapMode::CLAMP_TO_EDGE;
                params.wrapT = SamplerWrapMode::CLAMP_TO_EDGE;
                params.wrapR = SamplerWrapMode::CLAMP_TO_EDGE;
            }
            // GLES3.x specification forbids depth textures to be filtered.
            if (UTILS_UNLIKELY(isDepthFormat(t->format)
                               && params.compareMode == SamplerCompareMode::NONE
                               && params.filterMag != SamplerMagFilter::NEAREST
                               && params.filterMin != SamplerMinFilter::NEAREST
                               && params.filterMin != SamplerMinFilter::NEAREST_MIPMAP_NEAREST)) {
                params.filterMag = SamplerMagFilter::NEAREST;
                params.filterMin = SamplerMinFilter::NEAREST;
            }
            arg.t = t;
            arg.anisotropy = float(1u << params.anisotropyLog2);
            if constexpr (std::is_same_v<T, Sampler>) {
                arg.sampler = gl.getSampler(params);
            } else {
                arg.params = params;
            }
        }
    }, descriptors[binding].desc);
}

void GLDescriptorSet::bind(OpenGLContext& gl, OpenGLProgram* p, descriptor_set_t set,
        uint32_t const* offsets) const noexcept {
    assert_invariant(p);
    size_t dynamicOffsetIndex = 0;

    utils::bitset64 const activeDescriptorBindings = p->getActiveDescriptors(set);

    // loop only over the active indices for this program
    activeDescriptorBindings.forEachSetBit([this, &gl, p, set, offsets, &dynamicOffsetIndex]
            (size_t binding) {
        auto const& entry = descriptors[binding];
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Buffer>) {
                GLuint const bindingPoint = p->getBufferBinding(set, binding);
                GLintptr const offset = arg.offset;
                gl.bindBufferRange(arg.target, bindingPoint, arg.id, offset, arg.size);
            } else if constexpr (std::is_same_v<T, DynamicBuffer>) {
                GLuint const bindingPoint = p->getBufferBinding(set, binding);
                GLintptr const offset = arg.offset + offsets[dynamicOffsetIndex++];
                gl.bindBufferRange(arg.target, bindingPoint, arg.id, offset, arg.size);
            } else if constexpr (std::is_same_v<T, BufferGLES2>) {
                GLuint const bindingPoint = p->getBufferBinding(set, binding);
                GLintptr offset = arg.offset;
                if (arg.dynamicOffset) {
                    offset += offsets[dynamicOffsetIndex++];
                }
                gl.setEs2UniformBinding(bindingPoint,
                        arg.bo->gl.id,
                        static_cast<char const*>(arg.bo->gl.buffer) + offset,
                        arg.bo->age);
            } else if constexpr (std::is_same_v<T, Sampler>) {
                GLTexture const* const t = arg.t;
#if defined(GL_EXT_texture_filter_anisotropic)
                const bool anisotropyWorkaround =
                        gl.ext.EXT_texture_filter_anisotropic &&
                        gl.bugs.texture_filter_anisotropic_broken_on_sampler;
                if (UTILS_UNLIKELY(anisotropyWorkaround)) {
                    // Driver claims to support anisotropic filtering, but it fails when set on
                    // the sampler, we have to set it on the texture instead.
                    // The texture is already bound here.
                    glTexParameterf(t->gl.target, GL_TEXTURE_MAX_ANISOTROPY_EXT,
                            std::min(gl.gets.max_anisotropy, arg.anisotropy));
                }
#endif
                GLuint const unit = p->getTextureUnit(set, binding);
                gl.bindTexture(unit, t->gl.target, t->gl.id, t->gl.targetIndex);
                gl.bindSampler(unit, arg.sampler);

            } else if constexpr (std::is_same_v<T, SamplerGLES2>) {
                // in ES2 the sampler parameters need to be set on the texture itself
                GLTexture const* const t = arg.t;
                SamplerParams const params = arg.params;
                glTexParameteri(t->gl.target, GL_TEXTURE_MIN_FILTER,
                        (GLint)GLUtils::getTextureFilter(params.filterMin));
                glTexParameteri(t->gl.target, GL_TEXTURE_MAG_FILTER,
                        (GLint)GLUtils::getTextureFilter(params.filterMag));
                glTexParameteri(t->gl.target, GL_TEXTURE_WRAP_S,
                        (GLint)GLUtils::getWrapMode(params.wrapS));
                glTexParameteri(t->gl.target, GL_TEXTURE_WRAP_T,
                        (GLint)GLUtils::getWrapMode(params.wrapT));
#if defined(GL_EXT_texture_filter_anisotropic)
                glTexParameterf(t->gl.target, GL_TEXTURE_MAX_ANISOTROPY_EXT,
                        std::min(gl.gets.max_anisotropy, arg.anisotropy));
#endif
                assert_invariant(p);
                GLuint const unit = p->getTextureUnit(set, binding);
                gl.bindTexture(unit, t->gl.target, t->gl.id, t->gl.targetIndex);
            }
        }, entry.desc);
    });
    CHECK_GL_ERROR(utils::slog.e)
}

} // namespace filament::backend
