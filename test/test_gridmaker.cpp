#define BOOST_TEST_MODULE gridmaker_test
#include <boost/test/unit_test.hpp>
#include "test_util.h"
#include "grid_maker.h"
#include "example_extractor.h"
#include <iostream>

#define TOL 0.0001f
using namespace libmolgrid;

BOOST_AUTO_TEST_CASE(forward_cpu) {
  // hard-coded example, compared with a reference
  // read in example
  ExampleRef exref("1 ../../test/data/REC.pdb ../../test/data/LIG.mol", 1);
  std::shared_ptr<FileMappedGninaTyper> rectyper = 
    std::make_shared<FileMappedGninaTyper>("../../test/data/gnina35.recmap");
  std::shared_ptr<FileMappedGninaTyper> ligtyper = 
    std::make_shared<FileMappedGninaTyper>("../../test/data/gnina35.ligmap");
  ExampleProviderSettings settings;
  ExampleExtractor extractor(settings, rectyper, ligtyper);
  Example ex;
  extractor.extract(exref, ex);

  size_t ntypes = 0;
  size_t numsets = ex.sets.size();
  // for (size_t i=0; i<numsets; ++i) 
  for (size_t i=0; i<2; ++i) 
    ntypes += ex.sets[i].num_types();

  // set up gridmaker and run forward
  float dimension = 23.5;
  float resolution = 0.5;
  double half = dimension / 2.0;
  float3 grid_center = make_float3(-16.56986 + half, 0.63044 + half, -17.51435 + half);
  GridMaker gmaker(resolution, dimension);
  float3 grid_dims = gmaker.getGridDims();
  MGrid4f out(grid_dims.x, grid_dims.y, grid_dims.z, ntypes);
  Grid4f cpu_grid = out.cpu();
  //for some reason I'm getting three coordinate sets from the above...
  // for (size_t i=0; i<numsets; ++i) 
  for (size_t i=0; i<2; ++i) 
    gmaker.forward(grid_center, ex.sets[i], cpu_grid);

  // read in reference data
  std::ifstream ref("../../test/data/RECLIG.48.35.binmap", std::ios::binary);
  BOOST_CHECK_EQUAL((bool)ref, true);
  ref.seekg(0, ref.end);
  size_t N = ref.tellg();              
  ref.seekg(0, ref.beg);
  // size_t otherN = grid_dims.x * grid_dims.y * grid_dims.z * ntypes * sizeof(float);
  // std::cout << "N: " << N;
  // std::cout << " otherN: " << otherN;
  // std::cout << " ntypes: " << ntypes << "\n";
  std::vector<float> refdat(N / sizeof(float));
  ref.read(reinterpret_cast<char*>(refdat.data()), refdat.size()*sizeof(float));
  Grid4f ref_grid(refdat.data(), grid_dims.x, grid_dims.y, grid_dims.z, ntypes);

  // compare gridmaker result to reference
  for (size_t i=0; i<grid_dims.x; ++i) {
    for (size_t j=0; j<grid_dims.y; ++j) {
      for (size_t k=0; k<grid_dims.z; ++k) {
        for (size_t ch=0; ch<ntypes; ++ch) {
          BOOST_CHECK_SMALL(cpu_grid(i,j,k,ch) - ref_grid(i,j,k,ch), TOL);
        }
      }
    }
  }
}
