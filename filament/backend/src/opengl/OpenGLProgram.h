/*
 * Copyright (C) 2015 The Android Open Source Project
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

#ifndef TNT_FILAMENT_BACKEND_OPENGL_OPENGLPROGRAM_H
#define TNT_FILAMENT_BACKEND_OPENGL_OPENGLPROGRAM_H

#include "DriverBase.h"

#include "BindingMap.h"
#include "OpenGLContext.h"
#include "ShaderCompilerService.h"

#include <private/backend/Driver.h>

#include <backend/DriverEnums.h>
#include <backend/Program.h>

#include <utils/bitset.h>
#include <utils/compiler.h>
#include <utils/FixedCapacityVector.h>

#include <limits>

#include <stddef.h>
#include <stdint.h>

namespace filament::backend {

class OpenGLDriver;

class OpenGLProgram : public HwProgram {
public:

    OpenGLProgram() noexcept;
    OpenGLProgram(OpenGLDriver& gld, Program&& program) noexcept;
    ~OpenGLProgram() noexcept;

    bool isValid() const noexcept { return mToken || gl.program != 0; }

    void use(OpenGLDriver* const gld, OpenGLContext& context) noexcept {
        if (UTILS_UNLIKELY(!gl.program)) {
            initialize(*gld);
        }
        context.useProgram(gl.program);
    }

    GLuint getBufferBinding(descriptor_set_t set, descriptor_binding_t binding) const noexcept {
        return mBindingMap.get(set, binding);
    }

    GLuint getTextureUnit(descriptor_set_t set, descriptor_binding_t binding) const noexcept {
        return mBindingMap.get(set, binding);
    }

    utils::bitset64 getActiveDescriptors(descriptor_set_t set) const noexcept {
        return mBindingMap.getActiveDescriptors(set);
    }

    // For ES2 only
    void updateUniforms(uint32_t index, GLuint id, void const* buffer, uint16_t age) noexcept;
    void setRec709ColorSpace(bool rec709) const noexcept;

private:
    // keep these away from of other class attributes
    struct LazyInitializationData;

    void initialize(OpenGLDriver& gld);

    void initializeProgramState(OpenGLContext& context, GLuint program,
            LazyInitializationData& lazyInitializationData) noexcept;

    BindingMap mBindingMap;     // 8 bytes + out-of-line 256 bytes

    ShaderCompilerService::program_token_t mToken{};    // 16 bytes

    // only needed for ES2
    using LocationInfo = utils::FixedCapacityVector<GLint>;
    struct UniformsRecord {
        Program::UniformInfo uniforms;
        LocationInfo locations;
        mutable GLuint id = 0;
        mutable uint16_t age = std::numeric_limits<uint16_t>::max();
    };
    UniformsRecord const* mUniformsRecords = nullptr;
    GLint mRec709Location = -1; // 4 bytes

public:
    struct {
        GLuint program = 0;
    } gl;                                               // 4 bytes
};

// if OpenGLProgram is larger than 96 bytes, it'll fall in a larger Handle bucket.
static_assert(sizeof(OpenGLProgram) <= 96); // currently 96 bytes

} // namespace filament::backend

#endif // TNT_FILAMENT_BACKEND_OPENGL_OPENGLPROGRAM_H
