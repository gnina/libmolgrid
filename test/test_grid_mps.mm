/*
 * test_grid_mps.mm
 *
 * Tests for Grid<> on Apple Silicon (replaces test_grid.cu).
 * On Apple Silicon all memory is unified so GPU grids are accessible
 * directly from the CPU.
 */

#define BOOST_TEST_MODULE grid_mps_test
#include <boost/test/unit_test.hpp>

#include <numeric>
#include "libmolgrid/grid.h"

using namespace libmolgrid;

BOOST_AUTO_TEST_CASE( constructors )
{
  // Compilation + basic layout test.
  const int SIZE = 256 * 8;
  float  *f = (float*)  malloc(SIZE);
  double *d = (double*) malloc(SIZE);
  memset(f, 0, SIZE);
  memset(d, 0, SIZE);

  Grid1fCUDA g1f(f, 100);
  Grid2fCUDA g2f(f, 100, 1);
  Grid4fCUDA g4f(f, 2, 1, 2, 25);
  Grid6fCUDA g6f(f, 1, 2, 1, 2, 25, 1);

  Grid1dCUDA g1d(d, 3);
  Grid2dCUDA g2d(d, 64, 2);
  Grid4dCUDA g4d(d, 2, 2, 2, 16);

  BOOST_CHECK_EQUAL(g1f.data(), g6f.data());
  BOOST_CHECK_EQUAL(g2d.data(), g4d.data());
  BOOST_CHECK_NE((void*)g4f.data(), (void*)g4d.data());

  free(f);
  free(d);
}

BOOST_AUTO_TEST_CASE( direct_indexing )
{
  const int N = 256;
  // Unified memory: use malloc; accessible from Metal kernels.
  float  *f = (float*)  malloc(N * sizeof(float));
  double *d = (double*) malloc(N * sizeof(double));
  memset(f, 0, N * sizeof(float));
  memset(d, 0, N * sizeof(double));

  for (int i = 0; i < N; i++) f[i] = (float)i;

  Grid2fCUDA F(f, 8, 32);
  Grid2dCUDA D(d, 8, 32);

  // CPU copy (no kernel needed – this is the unified-memory equivalent).
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 32; j++)
      D(i, j) = F(i, j);

  for (int i = 0; i < N; i++) BOOST_CHECK_EQUAL(d[i], (double)i);

  free(f);
  free(d);
}

BOOST_AUTO_TEST_CASE( indirect_indexing )
{
  const int N = 256;
  float  *f = (float*)  malloc(N * sizeof(float));
  double *d = (double*) malloc(N * sizeof(double));
  for (int i = 0; i < N; i++) { f[i] = (float)i; d[i] = 0.0; }

  Grid3fCUDA F(f, 8, 16, 2);
  Grid3dCUDA D(d, 8, 16, 2);

  // CPU copy via bracket indexing
  for (int i = 0; i < 8;  i++)
    for (int j = 0; j < 16; j++)
      for (int k = 0; k < 2;  k++)
        D[i][j][k] = F[i][j][k];

  for (int i = 0; i < N; i++) BOOST_CHECK_EQUAL(d[i], (double)i);

  // Zero out slice F[3][5]
  Grid1fCUDA F1 = F[3][5];
  memset(F1.data(), 0, F1.size() * sizeof(float));

  float fsum = std::accumulate(f, f + N, 0.0f);
  BOOST_CHECK_EQUAL(fsum, 32427.0f);

  free(f);
  free(d);
}
