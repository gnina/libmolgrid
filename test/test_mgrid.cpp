/*
 * test_grid.cpp
 *
 *  Created on: Dec 20, 2018
 *      Author: dkoes
 */

#define BOOST_TEST_MODULE grid_test
#include <boost/test/unit_test.hpp>

#include "managed_grid.h"

using namespace libmolgrid;

BOOST_AUTO_TEST_CASE( constructors )
{
  //this is mostly a compilation test
  MGrid1f g1f(100);
  MGrid2f g2f(100, 1);
  MGrid4f g4f(2, 1, 2, 25);
  MGrid6f g6f(1,2,1,2,25,1);

  MGrid1d g1d(3);
  MGrid2d g2d(64,2);
  MGrid4d g4d(2,2,2,16);

  BOOST_CHECK_NE(g1f.data(),g6f.data());
  BOOST_CHECK_NE(g2d.data(),g4d.data());
  BOOST_CHECK_NE((void*)g4f.data(),(void*)g4d.data());
}

BOOST_AUTO_TEST_CASE( direct_indexing )
{

  MGrid4f g(2,4,8,4);
  MGrid4d gd(2,4,8,4);

  for(unsigned i = 0; i < 256; i++) {
    g.data()[i] = i;
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
    BOOST_CHECK_EQUAL(g.data()[i],i);
  }
}
