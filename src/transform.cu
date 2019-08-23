/*
 * \file transform.cu
 *
 *  CUDA implementations of Transform class.
 */

#include "libmolgrid/transform.h"

namespace libmolgrid {

//kernel for paralel transform on a single coordinate set
//note the importance of pass by value
template <typename Dtype, bool dotranslate>
__global__ void transform_forward_kernel(unsigned n, Quaternion Q, float3 center, float3 translate, Grid<Dtype, 2, true> in, Grid<Dtype, 2, true> out) {
  LMG_CUDA_KERNEL_LOOP(i, n) {
    Dtype x,y,z;
    x = in(i,0);
    y = in(i,1);
    z = in(i,2);
    float3 newpt = make_float3(0,0,0);
    if(dotranslate) {
      newpt = Q.transform(x,y,z, center, translate);
    } else {
      newpt = Q.transform(x,y,z, center, newpt);
    }

    out(i,0) = newpt.x;
    out(i,1) = newpt.y;
    out(i,2) = newpt.z;

  }
}

//kernel for translating coordinates
template <typename Dtype>
__global__ void transform_translate_kernel(unsigned n, float3 translate, Grid<Dtype, 2, true> in, Grid<Dtype, 2, true> out) {
  LMG_CUDA_KERNEL_LOOP(i, n) {
    Dtype x,y,z;
    x = in(i,0);
    y = in(i,1);
    z = in(i,2);
    out(i,0) = x+translate.x;
    out(i,1) = y+translate.y;
    out(i,2) = z+translate.z;
  }
}

//kernel for rotating coordinates about center
template <typename Dtype>
__global__ void transform_rotate_kernel(unsigned n, Quaternion Q, float3 center, Grid<Dtype, 2, true> in, Grid<Dtype, 2, true> out) {
  LMG_CUDA_KERNEL_LOOP(i, n) {
    Dtype x,y,z;
    x = in(i,0);
    y = in(i,1);
    z = in(i,2);
    float3 newpt = Q.rotate(x-center.x,y-center.y,z-center.z);

    out(i,0) = newpt.x+center.x;
    out(i,1) = newpt.y+center.y;
    out(i,2) = newpt.z+center.z;
  }
}



template <typename Dtype>
 __host__ void Transform::forward(const Grid<Dtype, 2, true>& in, Grid<Dtype, 2, true>& out, bool dotranslate) const {
  checkGrids(in,out);

  unsigned N = in.dimension(0);
  if(dotranslate)
    transform_forward_kernel<float, true><<<LMG_GET_BLOCKS(N), LMG_GET_THREADS(N)>>>(N, Q, center, translate, in, out);
  else
    transform_forward_kernel<float, false><<<LMG_GET_BLOCKS(N), LMG_GET_THREADS(N)>>>(N, Q, center, translate, in, out);
}

template __host__ void Transform::forward(const Grid<float, 2, true>&, Grid<float, 2, true>&, bool) const;

template <typename Dtype>
 __host__  void Transform::backward(const Grid<Dtype, 2, true>& in, Grid<Dtype, 2, true>& out, bool dotranslate) const {
  checkGrids(in,out);
  unsigned N = in.dimension(0);
  Quaternion invQ = Q.inverse();

  if(dotranslate) {
    float3 untranslate = make_float3(-translate.x,-translate.y,-translate.z);
    transform_translate_kernel<Dtype><<<LMG_GET_BLOCKS(N), LMG_GET_THREADS(N)>>>(N, untranslate, in, out);
    transform_rotate_kernel<Dtype><<<LMG_GET_BLOCKS(N), LMG_GET_THREADS(N)>>>(N,invQ,center,out,out);
  } else {
    transform_rotate_kernel<Dtype><<<LMG_GET_BLOCKS(N), LMG_GET_THREADS(N)>>>(N,invQ,center,in,out);
  }
  LMG_CUDA_CHECK(cudaPeekAtLastError());
}

template __host__ void Transform::backward(const Grid<float, 2, true>&, Grid<float, 2, true>&, bool) const;
template __host__ void Transform::backward(const Grid<double, 2, true>&, Grid<double, 2, true>&, bool) const;


}
