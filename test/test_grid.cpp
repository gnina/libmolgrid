/*
 * test_mgrid.cpp
 *
 *  Created on: Dec 20, 2018
 *      Author: dkoes
 */

#define BOOST_TEST_MODULE mgrid_test
#include <boost/test/unit_test.hpp>

#include <vector>
#include "libmolgrid/grid.h"

using namespace libmolgrid;


BOOST_AUTO_TEST_CASE( constructors )
{
  //this is mostly a compilation test
  float f[100] = {0,};
  Grid1f g1f(f, 100);
  Grid2f g2f(f, 100, 1);
  Grid4f g4f(f, 2, 1, 2, 25);
  Grid6f g6f(f, 1,2,1,2,25,1);

  double d[256] = {0,};
  Grid1d g1d(d,3);
  Grid2d g2d(d,64,2);
  Grid4d g4d(d,2,2,2,16);

  BOOST_CHECK_EQUAL(g1f.data(),g6f.data());
  BOOST_CHECK_EQUAL(g2d.data(),g4d.data());
  BOOST_CHECK_NE((void*)g4f.data(),(void*)g4d.data());
}

BOOST_AUTO_TEST_CASE( direct_indexing )
{
  float f[256] = {0,};
  double d[256] = {0,};
  for(unsigned i = 0; i < 256; i++) {
    f[i] = i;
  }

  Grid4f g(f,2,4,8,4);

  BOOST_CHECK_EQUAL(g(0,0,0,0), 0);
  BOOST_CHECK_EQUAL(g(1,1,1,1), 165);
  BOOST_CHECK_EQUAL(g(1,3,7,3),255);

  Grid4d gd(d,2,4,8,4);
  for(unsigned a = 0; a < 2; a++)
    for(unsigned b = 0; b < 4; b++)
      for(unsigned c = 0; c < 8; c++)
        for(unsigned d = 0; d < 4; d++) {
          gd(a,b,c,d) = g(a,b,c,d);
        }

  for(unsigned i = 0; i < 256; i++) {
    BOOST_CHECK_EQUAL(g.data()[i],i);
  }
}

BOOST_AUTO_TEST_CASE( indirect_indexing )
{
  float f[256] = {0,};
  for(unsigned i = 0; i < 256; i++) {
    f[i] = i;
  }

  Grid4f g(f,2,4,8,4);

  BOOST_CHECK_EQUAL(g[0][0][0][0], 0);
  BOOST_CHECK_EQUAL(g[1][1][1][1], 165);
  BOOST_CHECK_EQUAL(g[1][3][7][3],255);

  //should be able to slice whole subarray

  Grid3f h = g[1];
  BOOST_CHECK_EQUAL(h[0][0][0], 128);
  BOOST_CHECK_EQUAL(h[3][7][0],252);

  Grid2f g2 = h[3];
  BOOST_CHECK_EQUAL(g2[6][2], 250);

  g2[7][3] = 314;
  BOOST_CHECK_EQUAL(g[1][3][7][3],314);
}

BOOST_AUTO_TEST_CASE( blank_grid )
{
  std::vector<Grid2f> vec(2);
  float data[3][2] = {0};
  data[1][1] = 3;
  vec[0] = Grid2f(&data[0][0], 3, 2);
  BOOST_CHECK_EQUAL(vec[0][1][1],3.0);
  BOOST_CHECK_EQUAL(vec[1].dimension(0), 0);
  BOOST_CHECK_EQUAL(vec[1].dimension(1), 0);

  std::vector<Grid1f> vec1(3);
  BOOST_CHECK_EQUAL(vec1[0].dimension(0), 0);
}
