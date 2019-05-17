/*
 * test_grid.cu
 *
 *  Created on: Dec 20, 2018
 *      Author: dkoes
 */

#define BOOST_TEST_MODULE grid_cuda_test
#include <boost/test/unit_test.hpp>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "libmolgrid/managed_grid.h"

using namespace libmolgrid;


BOOST_AUTO_TEST_CASE( unifiedmem )
{
  //make sure data is accessible from both cpu and gpu
  MGrid1f g(100);
  for(unsigned i = 0; i < 100; i++) {
    g[i] = i;
  }

  float sum = thrust::reduce(thrust::device, g.gpu().data(), g.gpu().data()+g.size());
  BOOST_CHECK_EQUAL(sum, 4950);

  cudaError_t error = cudaGetLastError();
  BOOST_CHECK_EQUAL(error,cudaSuccess);
}



BOOST_AUTO_TEST_CASE( grid_conversion )
{
  //check conversion to cuda grid -- this should be faster, but don't know how to check that
  MGrid3f g3(7,13,11);
  MGrid1f g1(100);

  for(unsigned i = 0; i < 7; i++)
    for(unsigned j = 0; j < 13; j++)
      for(unsigned k = 0; k < 11; k++) {
        g3[i][j][k] = i+j+k;
      }

  for(unsigned i = 0; i < 100; i++) {
    g1(i) = i;
  }

  Grid3fCUDA gpu3(g3);
  Grid1fCUDA gpu1 = g1.gpu();

  float sum3 = thrust::reduce(thrust::device, gpu3.data(), gpu3.data()+gpu3.size());
  BOOST_CHECK_EQUAL(sum3,14014);

  float sum1 = thrust::reduce(thrust::device, gpu1.data(), gpu1.data()+gpu1.size());
  BOOST_CHECK_EQUAL(sum1,4950);

  cudaError_t error = cudaGetLastError();
  BOOST_CHECK_EQUAL(error,cudaSuccess);
}
