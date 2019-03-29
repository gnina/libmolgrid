#define BOOST_TEST_MODULE gridmaker_cuda_test
#include <boost/test/unit_test.hpp>
#include "grid_maker.h"
#include "atom_typer.h"
#include "test_util.h"
#include <iostream>
#include <iomanip>

#define TOL 0.0001f
using namespace libmolgrid;

BOOST_AUTO_TEST_CASE(forward_agreement) {
  size_t natoms = 1000;
  float resolution = 0.5; 
  float dimension = 23.5;
  float radiusmultiple = 1.5;
  GridMaker gmaker(resolution, dimension, radiusmultiple);
  float3 dim = gmaker.getGridDims();

  //randomly generated example, check equivalence between gpu and cpu versions
  random_engine.seed(0);
  MGrid2f coords(natoms, 3);
  MGrid1f type_indices(natoms); 
  MGrid1f radii(natoms);
  MGrid4f cout(dim.x, dim.y, dim.z, GninaIndexTyper::NumTypes);
  make_mol(coords.cpu(), type_indices.cpu(), radii.cpu(), natoms);
  float3 grid_center = make_float3(0,0,0); //coords generated from -25 to 25
                                           //so this should be ok

  //make grid
  std::fill(cout.data(), cout.data() + cout.size(), 0.0);
  gmaker.forward(grid_center, coords.cpu(), type_indices.cpu(), radii.cpu(), cout.cpu());

  Grid2fCUDA gcoords = coords.gpu();
  Grid1fCUDA gtype_indices = type_indices.gpu();
  Grid1fCUDA gradii = radii.gpu();
  size_t ntypes = GninaIndexTyper::NumTypes;
  size_t gsize = dim.x * dim.y * dim.z * ntypes;
  MGrid4f gout(dim.x, dim.y, dim.z, ntypes);
  LMG_CUDA_CHECK(cudaMemset(gout.data(), 0, gsize * sizeof(float)));
  gmaker.forward(grid_center, gcoords, gtype_indices, gradii, gout.gpu());
  cudaError_t error = cudaGetLastError();
  BOOST_CHECK_EQUAL(error, cudaSuccess);
  gout.tocpu();

  // std::ofstream out("out");
  // out.precision(5);
  // std::setprecision(5);
  //check equivalence
  for (size_t i=0; i<dim.x; ++i) {
    for (size_t j=0; j<dim.y; ++j) {
      for (size_t k=0; k<dim.z; ++k) {
        for (size_t ch=0; ch<GninaIndexTyper::NumTypes; ++ch) {
          // out << cout(i,j,k,ch);
          // out << " ";
          // out << gout(i,j,k,ch);
          // out << "\n";
          BOOST_CHECK_SMALL(cout(i,j,k,ch) - gout(i,j,k,ch), TOL);
        }
      }
    }
  }

  //check grid wasn't empty
  BOOST_CHECK_EQUAL(grid_empty(cout.cpu()), false);
  BOOST_CHECK_EQUAL(grid_empty(gout.cpu()), false);
}
