/*
 * test_grid.cpp
 *
 *  Created on: Dec 20, 2018
 *      Author: dkoes
 */

#define BOOST_TEST_MODULE grid_test
#include <boost/test/unit_test.hpp>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "libmolgrid/managed_grid.h"

using namespace libmolgrid;

BOOST_AUTO_TEST_CASE( constructors )
{
  //this is mostly a compilation test
  MGrid1f g1f(100);
  MGrid2f g2f(100, 1);
  MGrid4f g4f(2, 1, 2, 25);
  MGrid6f g6f(1,2,1,2,25,1);
  MGrid2f zero(0,0); //this should not fail

  MGrid1d g1d(3);
  MGrid2d g2d(64,2);
  MGrid4d g4d(2,2,2,16);

  BOOST_CHECK_NE(g1f.cpu().data(),g6f.cpu().data());
  BOOST_CHECK_NE(g2d.cpu().data(),g4d.cpu().data());
  BOOST_CHECK_NE((void*)g4f.cpu().data(),(void*)g4d.cpu().data());
}

BOOST_AUTO_TEST_CASE( direct_indexing )
{

  MGrid4f g(2,4,8,4);
  MGrid4d gd(2,4,8,4);

  for(unsigned i = 0; i < 256; i++) {
    g.cpu().data()[i] = i;
  }

  BOOST_CHECK_EQUAL(g(0,0,0,0), 0);
  BOOST_CHECK_EQUAL(g(1,1,1,1), 165);
  BOOST_CHECK_EQUAL(g(1,3,7,3),255);

  for(unsigned a = 0; a < 2; a++)
    for(unsigned b = 0; b < 4; b++)
      for(unsigned c = 0; c < 8; c++)
        for(unsigned d = 0; d < 4; d++) {
          gd(a,b,c,d) = g(a,b,c,d);
        }

  for(unsigned i = 0; i < 256; i++) {
    BOOST_CHECK_EQUAL(g.cpu().data()[i],i);
  }
}

BOOST_AUTO_TEST_CASE( indirect_indexing )
{
  MGrid4f g(2,4,8,4);
  for(unsigned i = 0; i < 256; i++) {
    g.cpu().data()[i] = i;
  }
  BOOST_CHECK_EQUAL(g[0][0][0][0], 0);
  BOOST_CHECK_EQUAL(g[1][1][1][1], 165);
  BOOST_CHECK_EQUAL(g[1][3][7][3],255);

  //should be able to slice whole subarray as views

  MGrid3f h = g[1];
  BOOST_CHECK_EQUAL(h[0][0][0], 128);
  BOOST_CHECK_EQUAL(h[3][7][0],252);

  MGrid2f g2 = h[3];
  BOOST_CHECK_EQUAL(g2[6][2], 250);

  g2[7][3] = 314;
  BOOST_CHECK_EQUAL(g[1][3][7][3],314);

  MGrid1f g1 = h[0][0];
  BOOST_CHECK_EQUAL(g1[0],128);

  MGrid4f gg = g;
  BOOST_CHECK_EQUAL(gg.cpu().data(),g.cpu().data());

}

BOOST_AUTO_TEST_CASE( grid_conversion )
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


BOOST_AUTO_TEST_CASE( blank_mgrid )
{
  std::vector<MGrid1f> vec(2);
  MGrid1f m(3);
  m[1] = 3.0;
  vec[0] = m;
  BOOST_CHECK_EQUAL(vec[0][1],3.0);
  BOOST_CHECK_EQUAL(vec[1].dimension(0), 0);

  std::vector<MGrid3f> vec3(3);
  BOOST_CHECK_EQUAL(vec3[1].dimension(0), 0);
  BOOST_CHECK_EQUAL(vec3[1].dimension(1), 0);
  BOOST_CHECK_EQUAL(vec3[1].dimension(2), 0);

}

BOOST_AUTO_TEST_CASE(resized)
{
  MGrid2f g(5,6);
  MGrid2f g2 = g.resized(4,6); //smaller, so should be same mem
  g(3,4) = 2.0;
  BOOST_CHECK_EQUAL(g2(3,4), 2.0);

  g2 = g2.resized(10,6); //larger, should be different mem with same values
  BOOST_CHECK_EQUAL(g2(3,4), 2.0);

  g(3,4) = 1.0;
  BOOST_CHECK_EQUAL(g2(3,4), 2.0); //should not change

  g2[9][5] = 3.0; //exists
  BOOST_CHECK_EQUAL(g2(9,5),3.0);

  BOOST_CHECK_EQUAL(g.size(), 30);
  BOOST_CHECK_EQUAL(g2.size(), 60);
}

BOOST_AUTO_TEST_CASE(copyto)
{
  MGrid2f g(3,2);
  MGrid2f h(2,2);
  g(1,1) = 3;
  g.copyTo(h);
  BOOST_CHECK_EQUAL(h(1,1),3);

  MGrid2f z;
  z.copyTo(h);  //dealing with empty and not overwritting extra
  BOOST_CHECK_EQUAL(h(1,1),3);

}
