/*
 * common.h
 *
 * Utility functions and definitions
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <memory>
#include <cuda_runtime_api.h>
#include <cuda.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


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


#endif /* COMMON_H_ */
