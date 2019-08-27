/*
 * coordinateset.cu
 *
 *  Created on: Aug 7, 2019
 *      Author: dkoes
 */
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include "libmolgrid/coordinateset.h"


namespace libmolgrid {

using namespace std;


//kernel parallelized over types
__global__ void sum_vector_types_gpu(Grid<float, 2, true> types, Grid<float, 1, true> sum) {
  unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  float tsum = 0.0;
  for(unsigned i = 0, n = types.dimension(0); i < n; i++) {
    tsum += types[i][t];
  }
  sum[t] = tsum;
}

//kernel parallelized over types - parrallelizing over index types this way is silly
//and likely ineffective, but I'm not expecting it to be used/matter much
// you at least get a parallel store, so that's something?
__global__ void sum_index_types_gpu(Grid<float, 1, true> types, Grid<float, 1, true> sum) {
  float t = blockIdx.x * blockDim.x + threadIdx.x;
  float tsum = 0.0;
  for(unsigned i = 0, n = types.dimension(0); i < n; i++) {
    if(types[i] == t)
      tsum += types[i];
  }
  sum[t] = tsum;
}

void CoordinateSet::sum_types(Grid<float, 1, true>& sum, bool zerofirst) const {
  if(zerofirst) sum.fill_zero();
  int NT = num_types();
  int blocks = LMG_GET_BLOCKS(NT);
  int threads = LMG_GET_THREADS(NT);
  if(!has_vector_types()) {
    sum_index_types_gpu<<<blocks,threads>>>(type_index.gpu(), sum);
  } else { //vector types
    sum_vector_types_gpu<<<blocks,threads>>>(type_vector.gpu(), sum);
  }
  LMG_CUDA_CHECK(cudaPeekAtLastError());

  //thrust::device_ptr<float> start = thrust::device_pointer_cast(sum.data());
  //return thrust::reduce(start, start+sum.size(), 0, thrust::plus<float>);
}

}
