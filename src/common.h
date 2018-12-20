/*
 * common.h
 *
 * Utility functions and definitions
 */

#ifndef COMMON_H_
#define COMMON_H_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


#endif /* COMMON_H_ */
