/*
 * test_transform_mps.mm
 *
 * Tests for Transform on Apple Silicon (replaces test_transform.cu).
 * The GPU path now uses Metal; behaviour is identical to the CUDA version.
 */

#define BOOST_TEST_MODULE transform_mps_test
#include <boost/test/unit_test.hpp>

#include "libmolgrid/transform.h"
#include "libmolgrid/managed_grid.h"
#include "test_transform.h"

using namespace libmolgrid;


BOOST_AUTO_TEST_CASE(apply_transform)
{
  // Non-random transform
  Quaternion q(sqrt(0.5f), 0, 0, sqrt(0.5f)); // 90-degree rotation around Z
  Transform nr(q, make_vec3(0,1,1), make_vec3(2,0,-3));

  // Random transform
  Transform r(make_vec3(0,1,1), 10.0f, true);

  float coord_data[8][3] = {
    {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1},
    {-1,.5f,3}, {1,1,1}, {0,1,1}, {.333f,.75f,-9}
  };
  float buffer[8][3] = {};

  MGrid2f coords(8, 3);
  for (unsigned i = 0; i < 8; i++)
    for (unsigned j = 0; j < 3; j++)
      coords[i][j] = coord_data[i][j];

  MGrid2f coords2(8, 3);

  // Apply non-random transform on GPU (Metal)
  nr.forward(coords.gpu(), coords2.gpu());

  Vec3 expected = make_vec3(2,1,-2);
  eqPt(coords2[6], expected); // at center

  expected = make_vec3(2,1,-3);
  eqPt(coords2[2], expected); // on z-axis

  expected = make_vec3(2,2,-2);
  eqPt(coords2[5], expected);

  // Input should be unchanged
  expected = make_vec3(0.333f, .75f, -9);
  eqPt(coords[7], expected);

  // Random transform – GPU output should differ from input
  r.forward(coords.gpu(), coords2.gpu());
  for (unsigned i = 0; i < 8; i++)
    neqPt(coords[i], coords2[i]);

  // CPU calculation should match GPU on unified memory
  Grid2f cpucoords(coords.cpu().data(), 8, 3);
  Grid2f cpucoords2((float*)buffer, 8, 3);
  r.forward(cpucoords, cpucoords2);
  for (unsigned i = 0; i < 8; i++)
    eqPt(cpucoords2[i], coords2[i]);

  // Backward should recover original
  r.backward(coords2.gpu(), coords2.gpu());
  coords2.tocpu();

  for (unsigned i = 0; i < 8; i++) {
    std::cerr << "(" << coords[i][0]  << "," << coords[i][1]  << "," << coords[i][2]  << ")  ("
              << coords2[i][0] << "," << coords2[i][1] << "," << coords2[i][2] << ")  ("
              << cpucoords2[i][0] << "," << cpucoords2[i][1] << "," << cpucoords2[i][2] << ")\n";
    eqPt(coords[i], coords2[i]);
  }
}
