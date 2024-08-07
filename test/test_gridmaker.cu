#define BOOST_TEST_MODULE gridmaker_cuda_test
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
  float3 dim = gmaker.get_grid_dims();

  //randomly generated example, check equivalence between gpu and cpu versions
  random_engine.seed(0);
  MGrid2f coords(natoms, 3);
  MGrid1f type_indices(natoms); 
  MGrid1f radii(natoms);
  size_t ntypes = (unsigned)GninaIndexTyper::NumTypes;
  MGrid4f cout(ntypes, dim.x, dim.y, dim.z);
  make_mol(coords.cpu(), type_indices.cpu(), radii.cpu(), natoms);
  float3 grid_center = make_float3(0,0,0); //coords generated from -25 to 25
                                           //so this should be ok

  //make grid
  gmaker.forward(grid_center, coords.cpu(), type_indices.cpu(), radii.cpu(), cout.cpu());

  Grid2fCUDA gcoords = coords.gpu();
  Grid1fCUDA gtype_indices = type_indices.gpu();
  Grid1fCUDA gradii = radii.gpu();
  MGrid4f gout(ntypes, dim.x, dim.y, dim.z);
  gmaker.forward(grid_center, gcoords, gtype_indices, gradii, gout.gpu());
  cudaError_t error = cudaGetLastError();
  BOOST_CHECK_EQUAL(error, cudaSuccess);
  gout.tocpu();

  //check equivalence
  for (size_t ch=0; ch<GninaIndexTyper::NumTypes; ++ch) {
    for (size_t i=0; i<dim.x; ++i) {
      for (size_t j=0; j<dim.y; ++j) {
        for (size_t k=0; k<dim.z; ++k) {
          size_t offset = ((((ch * dim.x) + i) * dim.y) + j) * dim.z + k;
          BOOST_CHECK_SMALL(*(cout.cpu().data()+offset) - *(gout.cpu().data()+offset), TOL);
        }
      }
    }
  }

  //check grid wasn't empty
  BOOST_CHECK_EQUAL(grid_empty(cout.cpu()), false);
  BOOST_CHECK_EQUAL(grid_empty(gout.cpu()), false);
}

