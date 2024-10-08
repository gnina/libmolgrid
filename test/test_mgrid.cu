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

BOOST_AUTO_TEST_CASE( grid_conversion2 )
{
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

  Grid3f cpu3(g3);
  Grid1f cpu1 = g1.cpu();

  float sum3 = thrust::reduce(thrust::host, cpu3.data(), cpu3.data()+cpu3.size());
  BOOST_CHECK_EQUAL(sum3,14014);

  float sum1 = thrust::reduce(thrust::host, cpu1.data(), cpu1.data()+cpu1.size());
  BOOST_CHECK_EQUAL(sum1,4950);

  MGrid6d g6(3,4,5,2,1,10);
  g6[2][2][2][0][0][5] = 3.14;
  Grid6d cpu6 = (Grid6d)g6; //cast conversion
  BOOST_CHECK_EQUAL(cpu6.size(),1200);
  BOOST_CHECK_EQUAL(cpu6(2,2,2,0,0,5), 3.14);

  Grid6dCUDA gpu6 = (Grid6dCUDA)g6;
  double *cudaptr = gpu6.address(2,2,2,0,0,5);
  double val = 0;
  cudaMemcpy(&val, cudaptr, sizeof(double), cudaMemcpyDeviceToHost);
  BOOST_CHECK_EQUAL(val, 3.14);

}
