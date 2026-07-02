/*
 * test_mgrid_mps.mm
 *
 * Tests for ManagedGrid on Apple Silicon (replaces test_mgrid.cu).
 * On Apple Silicon unified memory the same buffer is visible to both
 * CPU and Metal GPU without explicit copies.
 */

#define BOOST_TEST_MODULE mgrid_mps_test
#include <boost/test/unit_test.hpp>

#include <numeric>
#include "libmolgrid/managed_grid.h"

using namespace libmolgrid;

BOOST_AUTO_TEST_CASE( unifiedmem )
{
  MGrid1f g(100);
  for (unsigned i = 0; i < 100; i++) g[i] = (float)i;

  // Unified memory: GPU view points at the same buffer.
  float sum = std::accumulate(g.gpu().data(),
                               g.gpu().data() + g.size(), 0.0f);
  BOOST_CHECK_EQUAL(sum, 4950.0f);
}

BOOST_AUTO_TEST_CASE( grid_conversion )
{
  MGrid3f g3(7, 13, 11);
  MGrid1f g1(100);

  for (unsigned i = 0; i < 7;   i++)
    for (unsigned j = 0; j < 13; j++)
      for (unsigned k = 0; k < 11; k++)
        g3[i][j][k] = (float)(i + j + k);

  for (unsigned i = 0; i < 100; i++) g1(i) = (float)i;

  Grid3fCUDA gpu3(g3);
  Grid1fCUDA gpu1 = g1.gpu();

  float sum3 = std::accumulate(gpu3.data(), gpu3.data() + gpu3.size(), 0.0f);
  BOOST_CHECK_EQUAL(sum3, 14014.0f);

  float sum1 = std::accumulate(gpu1.data(), gpu1.data() + gpu1.size(), 0.0f);
  BOOST_CHECK_EQUAL(sum1, 4950.0f);
}

BOOST_AUTO_TEST_CASE( grid_conversion2 )
{
  MGrid3f g3(7, 13, 11);
  MGrid1f g1(100);

  for (unsigned i = 0; i < 7;   i++)
    for (unsigned j = 0; j < 13; j++)
      for (unsigned k = 0; k < 11; k++)
        g3[i][j][k] = (float)(i + j + k);

  for (unsigned i = 0; i < 100; i++) g1(i) = (float)i;

  Grid3f cpu3(g3);
  Grid1f cpu1 = g1.cpu();

  float sum3 = std::accumulate(cpu3.data(), cpu3.data() + cpu3.size(), 0.0f);
  BOOST_CHECK_EQUAL(sum3, 14014.0f);

  float sum1 = std::accumulate(cpu1.data(), cpu1.data() + cpu1.size(), 0.0f);
  BOOST_CHECK_EQUAL(sum1, 4950.0f);

  MGrid6d g6(3, 4, 5, 2, 1, 10);
  g6[2][2][2][0][0][5] = 3.14;
  Grid6d cpu6 = (Grid6d)g6;
  BOOST_CHECK_EQUAL(cpu6.size(), 1200u);
  BOOST_CHECK_EQUAL(cpu6(2, 2, 2, 0, 0, 5), 3.14);

  // GPU view is same memory on Apple Silicon.
  Grid6dCUDA gpu6 = (Grid6dCUDA)g6;
  BOOST_CHECK_EQUAL(gpu6(2, 2, 2, 0, 0, 5), 3.14);
}
