#define BOOST_TEST_MODULE gridmaker_test
#include <boost/test/unit_test.hpp>
#include "test_util.h"
#include "grid_maker.h"
#include "example_extractor.h"
#include <iostream>
#include <iomanip>

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
  CoordinateSet combined = ex.merge_coordinates();

  size_t ntypes = combined.num_types();

  // set up gridmaker and run forward
  float dimension = 23.5;
  float resolution = 0.5;
  double half = dimension / 2.0;
  float3 grid_center = make_float3(-16.56986 + half, 0.63044 + half, -17.51435 + half);
  float3 grid_origin = make_float3(-16.56986, 0.63044, -17.51435);
  GridMaker gmaker(resolution, dimension);
  float3 grid_dims = gmaker.getGridDims();
  MGrid4f out(grid_dims.x, grid_dims.y, grid_dims.z, ntypes);
  Grid4f cpu_grid = out.cpu();
  gmaker.forward(grid_center, combined, cpu_grid);

  // read in reference data
  std::vector<float> refdat;
  std::ifstream ref("../../test/data/RECLIG.48.35.binmap");
  BOOST_CHECK_EQUAL((bool)ref, true);
  while(ref && ref.peek() != EOF) {
    float nextval = 0;
    ref.read((char*)&nextval, sizeof(float));
    refdat.push_back(nextval);
  }
  Grid4f ref_grid(refdat.data(), ntypes, grid_dims.x, grid_dims.y, grid_dims.z);

  std::setprecision(5);
  // compare gridmaker result to reference
  for (size_t ch=0; ch<ntypes; ++ch) {
    std::string fname = "ref_" + std::to_string(ch) + ".dx";
    std::string cname = "cpu_" + std::to_string(ch) + ".dx";
    std::ofstream fout(fname.c_str());
    std::ofstream cout(cname.c_str());
    write_dx_header(fout, grid_dims.x, grid_origin, resolution);
    write_dx_header(cout, grid_dims.x, grid_origin, resolution);
    unsigned total = 0;
    for (size_t i=0; i<grid_dims.x; ++i) {
      for (size_t j=0; j<grid_dims.y; ++j) {
        for (size_t k=0; k<grid_dims.z; ++k) {
          size_t offset = ((((ch * grid_dims.x) + i) * grid_dims.y) + j) * grid_dims.z + k;
          fout << *(ref_grid.data() + offset);
          cout << *(cpu_grid.data() + offset);
          total++;
          if (total % 3 == 0) {
            fout << "\n";
            cout << "\n";
          }
          else {
            fout << " ";
            cout << " ";
          }
          BOOST_CHECK_SMALL(*(cpu_grid.data()+offset) - *(ref_grid.data()+offset), TOL);
        }
      }
    }
  }
}
