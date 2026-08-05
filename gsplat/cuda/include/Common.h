/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>

// torch's hipify rewrites std::min/std::max to the global ::min/::max (the CUDA
// device spelling). ROCm 7.2's Clang does not expose those CUDA builtins in
// either pass, so keep the compatibility overloads in the global namespace
// where hipify expects to find them. CUDA is unaffected (USE_ROCM undefined).
#if defined(USE_ROCM)
template <typename T, typename U>
__host__ __device__ constexpr auto min(T a, U b) {
    return b < a ? b : a;
}
template <typename T, typename U>
__host__ __device__ constexpr auto max(T a, U b) {
    return a < b ? b : a;
}
#endif

namespace gsplat {

// ROCm 7.2's Clang device pass does not expose CUDA's unqualified min/max and
// reciprocal-square-root overloads when PyTorch extensions are built through
// the compiler driver. Keep those CUDA-compatible spellings local to gsplat
// and lower the floating-point operations directly to Clang builtins.
#if defined(USE_ROCM)
template <typename T, typename U>
__host__ __device__ constexpr auto min(T a, U b)
{
    return b < a ? b : a;
}

template <typename T, typename U>
__host__ __device__ constexpr auto max(T a, U b)
{
    return a < b ? b : a;
}

__host__ __device__ inline float fminf(float a, float b)
{
    return b < a ? b : a;
}

__host__ __device__ inline float sqrt(float x)
{
    return __builtin_sqrtf(x);
}

__host__ __device__ inline float sqrtf(float x)
{
    return __builtin_sqrtf(x);
}

__host__ __device__ inline float rsqrt(float x)
{
    return 1.0f / __builtin_sqrtf(x);
}

__host__ __device__ inline float rsqrtf(float x)
{
    return 1.0f / __builtin_sqrtf(x);
}

__host__ __device__ inline float __logf(float x)
{
    return __builtin_logf(x);
}

__host__ __device__ inline float log2f(float x)
{
    return __builtin_log2f(x);
}

__host__ __device__ inline float floor(float x)
{
    return __builtin_floorf(x);
}

__host__ __device__ inline float floorf(float x)
{
    return __builtin_floorf(x);
}

__host__ __device__ inline float ceil(float x)
{
    return __builtin_ceilf(x);
}

__host__ __device__ inline float ceilf(float x)
{
    return __builtin_ceilf(x);
}

__host__ __device__ inline float gsplat_atan2(float y, float x)
{
    return __builtin_atan2f(y, x);
}

__host__ __device__ inline float gsplat_fmod(float numerator, float denominator)
{
    return __builtin_fmodf(numerator, denominator);
}

__device__ inline float gsplat_divide_rn(float numerator, float denominator)
{
    // A named wrapper avoids the CUDA-only __fdiv_rn intrinsic. Keep this one
    // division precise even when a caller explicitly opts into FAST_MATH=1.
#pragma clang fp reciprocal(off)
    return numerator / denominator;
}
#else
__host__ __device__ inline float gsplat_atan2(float y, float x)
{
    return atan2f(y, x);
}

__host__ __device__ inline float gsplat_fmod(float numerator, float denominator)
{
    return fmodf(numerator, denominator);
}

__device__ inline float gsplat_divide_rn(float numerator, float denominator)
{
    return __fdiv_rn(numerator, denominator);
}
#endif

//
// Some Macros.
//
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten)                                                     \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...)                                                 \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)

//
// Convenience typedefs for CUDA types
//
using vec2 = glm::vec<2, float>;
using vec3 = glm::vec<3, float>;
using vec4 = glm::vec<4, float>;
using mat2 = glm::mat<2, 2, float>;
using mat3 = glm::mat<3, 3, float>;
using mat4 = glm::mat<4, 4, float>;
using mat3x2 = glm::mat<3, 2, float>;

//
// Legacy Camera Types
//
enum CameraModelType {
    PINHOLE = 0,
    ORTHO = 1,
    FISHEYE = 2,
    FTHETA = 3,
    LIDAR = 4,
};

#define N_THREADS_PACKED 256
#define ALPHA_THRESHOLD (1.f / 255.f)
// GAUSSIAN_EXTEND determines where the gaussian is truncated in standard deviations."
#define GAUSSIAN_EXTEND 3.33f
// MAX_ALPHA and TRANSMITTANCE_THRESHOLD are chosen so that the equivalent of
// a maximal opacity Gaussian has to be rasterized twice to reach the threshold,
// without getting the transmittance too small for numerical stability of
// the backward pass.
// i.e. TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2
#define MAX_ALPHA 0.99f
#define TRANSMITTANCE_THRESHOLD 1e-4f

#define MAX_KERNEL_DENSITY_CUTOFF 0.0113

// Floor for the antialiased compensation factor (sqrt(det_orig / det_blur)).
// Prevents compensation from reaching zero for extremely small Gaussians.
#define MIN_COMPENSATION 0.005f

// Floor for (1 - alpha) when computing 1/(1-alpha) in backward rasterization.
// Prevents gradient explosion when alpha approaches 1.0.
#define MIN_ONE_MINUS_ALPHA 1e-6f

#ifdef __CUDACC__
#   define GSPLAT_NOINLINE __noinline__
#else
#   define GSPLAT_NOINLINE
#endif

} // namespace gsplat
