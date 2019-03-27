/*
 * common.h
 *
 * Utility functions and definitions
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <memory>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cuda.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_DEVICE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_DEVICE_MEMBER
#endif

//called in device code to perform a parallel operation
#define LMG_CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
#define LMG_CUDA_NUM_THREADS 512
#define LMG_CUDA_BLOCKDIM 8
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

// CUDA: number of blocks for threads.
#define LMG_GET_BLOCKS(N) \
  ((N + LMG_CUDA_NUM_THREADS - 1) / LMG_CUDA_NUM_THREADS)

#ifndef __CUDA_ARCH__
#define LMG_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error != cudaSuccess) {                                          \
        std::cerr << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error); abort(); } \
  } while (0)
#else
// probably don't want to make API calls on the device.
#define LMG_CUDA_CHECK(condition) condition
#endif


namespace libmolgrid {
  // helper function for creating unified memory shared pointer with fallback to
  // host memory; sz is number of elements, memory is returned zeroed out
  template<typename Dtype>
  inline std::shared_ptr<Dtype> create_unified_shared_ptr(size_t sz) {
    std::shared_ptr<Dtype> ptr;
    Dtype *buffer = nullptr;
    cudaError_t err = cudaMallocManaged((void**)&buffer,sz*sizeof(Dtype));
    if(err != cudaSuccess) {
      //fallback on host memory
      buffer = (Dtype*)malloc(sz*sizeof(Dtype));
      ptr = std::shared_ptr<Dtype>(buffer);
    } else {
      //success, need to deallocate with cuda
      ptr = std::shared_ptr<Dtype>(buffer,cudaFree);
    }
    //zero out
    memset(buffer, 0, sz*sizeof(Dtype));
    return ptr;
  }


}
#endif /* COMMON_H_ */
