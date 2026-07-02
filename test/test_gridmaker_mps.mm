/*
 * test_gridmaker_mps.mm
 *
 * Tests for GridMaker on Apple Silicon (replaces test_gridmaker.cu).
 * On Apple Silicon unified memory the GPU and CPU share the same buffer,
 * so GPU and CPU forward passes must produce identical results.
 */

#define BOOST_TEST_MODULE gridmaker_mps_test
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>

#include "test_util.h"

#include "libmolgrid/grid_maker.h"
#include "libmolgrid/atom_typer.h"

#define TOL 0.0001f
using namespace libmolgrid;


BOOST_AUTO_TEST_CASE(forward_agreement) {
  size_t natoms = 1000;
  float resolution = 0.5;
  float dimension = 23.5;
  float radiusmultiple = 1.5;
  GridMaker gmaker(resolution, dimension, radiusmultiple);
  Vec3 dim = gmaker.get_grid_dims();

  // Randomly generated example: check equivalence between GPU (Metal) and CPU.
  random_engine.seed(0);
  MGrid2f coords(natoms, 3);
  MGrid1f type_indices(natoms);
  MGrid1f radii(natoms);
  size_t ntypes = (unsigned)GninaIndexTyper::NumTypes;
  MGrid4f cout(ntypes, dim.x, dim.y, dim.z);
  make_mol(coords.cpu(), type_indices.cpu(), radii.cpu(), natoms);
  Vec3 grid_center = make_vec3(0, 0, 0);

  // CPU forward pass
  gmaker.forward(grid_center, coords.cpu(), type_indices.cpu(), radii.cpu(), cout.cpu());

  // GPU (Metal) forward pass — same pointers on unified memory
  MGrid4f gout(ntypes, dim.x, dim.y, dim.z);
  gmaker.forward(grid_center, coords.gpu(), type_indices.gpu(), radii.gpu(), gout.gpu());
  // tocpu() is a no-op on unified memory but kept for API symmetry
  gout.tocpu();

  // Check equivalence
  for (size_t ch = 0; ch < GninaIndexTyper::NumTypes; ++ch) {
    for (size_t i = 0; i < dim.x; ++i) {
      for (size_t j = 0; j < dim.y; ++j) {
        for (size_t k = 0; k < dim.z; ++k) {
          size_t offset = ((((ch * dim.x) + i) * dim.y) + j) * dim.z + k;
          BOOST_CHECK_SMALL(*(cout.cpu().data() + offset) - *(gout.cpu().data() + offset), TOL);
        }
      }
    }
  }

  // Ensure grids are non-empty
  BOOST_CHECK_EQUAL(grid_empty(cout.cpu()), false);
  BOOST_CHECK_EQUAL(grid_empty(gout.cpu()), false);
}
