/*
 * common.h
 *
 * Utility functions and definitions.
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <memory>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <string>

#include "libmolgrid/config.h"

#if LIBMOLGRID_USE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

#if LIBMOLGRID_USE_CUDA && defined(__CUDACC__)
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_DEVICE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_DEVICE_MEMBER
#endif

// Backend-neutral vector types for the public/host API. These are plain PODs
// (not CUDA's own vector_types.h float3 etc.) so the API doesn't masquerade
// as CUDA-specific when built without CUDA. They're also valid __host__
// __device__ types under nvcc, so CUDA_CALLABLE_MEMBER functions can use them.
struct Vec3 { float x, y, z; };
struct UVec2 { unsigned int x, y; };

CUDA_CALLABLE_MEMBER inline Vec3 make_vec3(float x, float y, float z) { return {x, y, z}; }
CUDA_CALLABLE_MEMBER inline UVec2 make_uvec2(unsigned x, unsigned y) { return {x, y}; }

#if LIBMOLGRID_USE_CUDA
// Conversion at the host/device kernel-launch boundary: CUDA kernels still
// take native ::float3 (the real, hardware-recognized CUDA vector type).
// Built via aggregate init rather than make_float3() so this compiles from
// plain .cpp translation units too, not just .cu ones: make_float3 lives in
// <vector_functions.h>, which nvcc implicitly includes for .cu files but
// which isn't pulled in here for a host-only g++ compile.
inline ::float3 to_cuda(Vec3 v) { return ::float3{v.x, v.y, v.z}; }
#endif

#if LIBMOLGRID_USE_CUDA
#define LMG_CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#else
#define LMG_CUDA_KERNEL_LOOP(i, n) \
  for (int i = 0; i < static_cast<int>(n); i++)
#endif

#define LMG_CUDA_NUM_THREADS 512
#define LMG_CUDA_BLOCKDIM 8
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

#define LMG_GET_BLOCKS(N) ((unsigned(N) + LMG_CUDA_NUM_THREADS - 1) / LMG_CUDA_NUM_THREADS)
#define LMG_GET_THREADS(N) ((unsigned(N) < LMG_CUDA_NUM_THREADS) ? unsigned(N) : LMG_CUDA_NUM_THREADS)

#if LIBMOLGRID_USE_CUDA && !defined(__CUDA_ARCH__)
#define LMG_CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if(error != cudaSuccess) { \
      std::cerr << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error); \
      throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)
#elif LIBMOLGRID_USE_CUDA
#define LMG_CUDA_CHECK(condition) condition
#else
#define LMG_CUDA_CHECK(condition) (condition)
#endif

inline std::string itoa(size_t v) { return std::to_string(v); }

#endif /* COMMON_H_ */
