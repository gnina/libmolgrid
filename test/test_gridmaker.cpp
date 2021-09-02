#define BOOST_TEST_MODULE gridmaker_test
#include <boost/test/unit_test.hpp>
#include "test_util.h"
#include "libmolgrid/grid_maker.h"
#include "libmolgrid/example_extractor.h"
#include <iostream>
#include <iomanip>

#define TOL 0.0001f
using namespace libmolgrid;

BOOST_AUTO_TEST_CASE(forward_cpu) {
  // hard-coded example, compared with a reference
  // read in example
  ExampleRef exref("1 ../../test/data/REC.pdb ../../test/data/LIG.mol", 1);
  std::shared_ptr<FileMappedGninaTyper> rectyper =
      std::make_shared < FileMappedGninaTyper
          > ("../../test/data/gnina35.recmap");
  std::shared_ptr<FileMappedGninaTyper> ligtyper =
      std::make_shared < FileMappedGninaTyper
          > ("../../test/data/gnina35.ligmap");
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
  float3 grid_center = make_float3(-16.56986 + half, 0.63044 + half,
      -17.51435 + half);
  // float3 grid_origin = make_float3(-16.56986, 0.63044, -17.51435);
  GridMaker gmaker(resolution, dimension);
  float3 grid_dims = gmaker.get_grid_dims();
  MGrid4f out(ntypes, grid_dims.x, grid_dims.y, grid_dims.z);
  Grid4f cpu_grid = out.cpu();
  gmaker.forward(grid_center, combined, cpu_grid);

  // read in reference data
  std::vector<float> refdat;
  std::ifstream ref("../../test/data/RECLIG.48.35.binmap");
  BOOST_CHECK_EQUAL((bool )ref, true);
  while (ref && ref.peek() != EOF) {
    float nextval = 0;
    ref.read((char*) &nextval, sizeof(float));
    refdat.push_back(nextval);
  }
  Grid4f ref_grid(refdat.data(), ntypes, grid_dims.x, grid_dims.y, grid_dims.z);

  // std::setprecision(5);
  // compare gridmaker result to reference
  for (size_t ch = 0; ch < ntypes; ++ch) {
    // std::string fname = "ref_" + std::to_string(ch) + ".dx";
    // std::string cname = "cpu_" + std::to_string(ch) + ".dx";
    // std::ofstream fout(fname.c_str());
    // std::ofstream cout(cname.c_str());
    // write_dx_header(fout, grid_dims.x, grid_origin, resolution);
    // write_dx_header(cout, grid_dims.x, grid_origin, resolution);
    // unsigned total = 0;
    for (size_t i = 0; i < grid_dims.x; ++i) {
      for (size_t j = 0; j < grid_dims.y; ++j) {
        for (size_t k = 0; k < grid_dims.z; ++k) {
          size_t offset = ((((ch * grid_dims.x) + i) * grid_dims.y) + j)
              * grid_dims.z + k;
          // fout << *(ref_grid.data() + offset);
          // cout << *(cpu_grid.data() + offset);
          // total++;
          // if (total % 3 == 0) {
          // fout << "\n";
          // cout << "\n";
          // }
          // else {
          // fout << " ";
          // cout << " ";
          // }
          BOOST_CHECK_SMALL(
              *(cpu_grid.data() + offset) - *(ref_grid.data() + offset), TOL);
        }
      }
    }
  }

  //now try a different resolution
  dimension = 23.5;
  resolution = 0.25;
  grid_center = combined.center();
  GridMaker biggmaker(resolution, dimension);
  grid_dims = biggmaker.get_grid_dims();

  MGrid4f cout(ntypes, grid_dims.x, grid_dims.y, grid_dims.z);
  MGrid4f gout(ntypes, grid_dims.x, grid_dims.y, grid_dims.z);
  biggmaker.forward(grid_center, combined, cout.cpu());
  biggmaker.forward(grid_center, combined, gout.gpu());

  //compare
  gout.tocpu();
  for (size_t ch = 0; ch < ntypes; ++ch) {
    for (size_t i = 0; i < grid_dims.x; ++i) {
      for (size_t j = 0; j < grid_dims.y; ++j) {
        for (size_t k = 0; k < grid_dims.z; ++k) {
            if(::fabs(cout(ch,i,j,k) - gout(ch,i,j,k)) > 0.001)
              printf("%d %d,%d,%d   %f %f %f\n",int(ch),int(i),int(j),int(k),cout(ch,i,j,k),gout(ch,i,j,k),cout(ch,i,j,k) - gout(ch,i,j,k));
          BOOST_CHECK_SMALL(cout(ch,i,j,k) - gout(ch,i,j,k), TOL);
        }
      }
    }
  }
}

//boost assert equality between to sets of coordinates
static void same_coords(MGrid2f& a, MGrid2f& b) {
  BOOST_CHECK_EQUAL(a.dimension(0),b.dimension(0));
  BOOST_CHECK_EQUAL(a.dimension(1),b.dimension(1)); //should be 3 for coords
  for(unsigned i = 0, n = a.dimension(0); i < n; i++) {
    for (unsigned j = 0; j < a.dimension(1); j++) {
      BOOST_CHECK_SMALL(a(i,j) - b(i,j), TOL);
    }
  }
}

//compare two grids
static void same_grids(MGrid4f& a, MGrid4f& b) {
  BOOST_CHECK_EQUAL(a.dimension(0),b.dimension(0));
  BOOST_CHECK_EQUAL(a.dimension(1),b.dimension(1));
  BOOST_CHECK_EQUAL(a.dimension(2),b.dimension(2));
  BOOST_CHECK_EQUAL(a.dimension(3),b.dimension(3));
  a.tocpu();
  b.tocpu();
  for(unsigned i = 0, n = a.dimension(0); i < n; i++) {
    for (unsigned j = 0; j < a.dimension(1); j++) {
      for (unsigned k = 0; k < a.dimension(2); k++) {
        for (unsigned m = 0; m < a.dimension(3); m++) {
          if(fabs(a(i,j,k,m) - b(i,j,k,m)) > TOL)
            std::cerr << i << "," << j << "," << k << "," << m << "  " << a(i,j,k,m) << " " << b(i,j,k,m) << "\n";
          BOOST_CHECK_SMALL(a(i,j,k,m) - b(i,j,k,m), TOL);
        }
      }
    }
  }
}


BOOST_AUTO_TEST_CASE(backward) {
  using namespace std;
  GridMaker g(0.1, 6.0);

  vector<float3> c { make_float3(0, 0, 0) };
  vector<int> t { 0 };
  vector<float> r { 2.0 };

  CoordinateSet coords(c, t, r, 1);
  float dim = g.get_grid_dims().x;
  MGrid4f diff(1, dim, dim, dim);
  diff(0, 30, 30, 30) = 1.0;

  MGrid2f cpuatoms(1, 3);
  MGrid2f gpuatoms(1, 3);

  g.backward(float3 { 0, 0, 0 }, coords, diff.cpu(), cpuatoms.cpu());

  for (unsigned i = 0; i < 3; i++) {
    BOOST_CHECK_SMALL(cpuatoms(0, i), TOL);
  }

  g.backward(float3 { 0, 0, 0 }, coords, diff.gpu(), gpuatoms.gpu());
  same_coords(cpuatoms,gpuatoms);

  //move coordinate
  coords.coords(0, 0) = 1.0;

  g.backward(float3 { 0, 0, 0 }, coords, diff.cpu(), cpuatoms.cpu());

  float gval = cpuatoms(0, 0);
  BOOST_CHECK_LT(gval, -TOL);
  for (unsigned i = 1; i < 3; i++) { //first dimension should say move left
    BOOST_CHECK_SMALL(cpuatoms(0, i), TOL);
  }

  g.backward(float3 { 0, 0, 0 }, coords, diff.gpu(), gpuatoms.gpu());
  same_coords(cpuatoms,gpuatoms);

  //move to other side
  coords.coords(0, 0) = -1.0;
  g.backward(float3 { 0, 0, 0 }, coords, diff.cpu(), cpuatoms.cpu());

  BOOST_CHECK_GT(cpuatoms(0,0),TOL);
  BOOST_CHECK_SMALL(gval + cpuatoms(0, 0), TOL); //should be symmetric
  for (unsigned i = 1; i < 3; i++) {
    BOOST_CHECK_SMALL(cpuatoms(0, i), TOL);
  }

  g.backward(float3 { 0, 0, 0 }, coords, diff.gpu(), gpuatoms.gpu());
  same_coords(cpuatoms,gpuatoms);


  //does transform backwards work?
  Transform T(float3{0,0,0}, 0, true);
  T.forward(coords, coords);

  g.backward(float3 { 0, 0, 0 }, coords, diff.cpu(), cpuatoms.cpu());
  g.backward(float3 { 0, 0, 0 }, coords, diff.gpu(), gpuatoms.gpu());
  same_coords(cpuatoms,gpuatoms);

  //with random rotation, all three coordinates should have gradient
  for (unsigned i = 0; i < 3; i++) {
    BOOST_CHECK_GT(fabs(cpuatoms(0, i)), TOL);
  }

  T.backward(cpuatoms.cpu(), cpuatoms.cpu(), false);
  T.backward(gpuatoms.gpu(), gpuatoms.gpu(), false);
  same_coords(cpuatoms,gpuatoms);

  //should have positive x and zero y,z
  BOOST_CHECK_GT(cpuatoms(0,0),TOL);
  for (unsigned i = 1; i < 3; i++) {
    BOOST_CHECK_SMALL(cpuatoms(0, i), TOL);
  }
}

BOOST_AUTO_TEST_CASE(backward_relevance) {
  using namespace std;
  GridMaker g(0.1, 6.0);

  vector<float3> c { make_float3(0, 0, 0) };
  vector<int> t { 0 };
  vector<float> r { 2.0 };

  CoordinateSet coords(c, t, r, 1);
  float dim = g.get_grid_dims().x;
  MGrid4f diff(1, dim, dim, dim);
  diff(0, 31, 30, 30) = 10.0;

  MGrid4f density(1, dim, dim, dim);
  density(0, 31,30, 30) = 1.0; //offset so only partial relevance shoudl be propped

  MGrid1f cpurel(1);
  MGrid1f gpurel(1);

  g.backward_relevance(float3 { 0, 0, 0 }, coords, density.cpu(), diff.cpu(), cpurel.cpu());
  g.backward_relevance(float3 { 0, 0, 0 }, coords, density.gpu(), diff.gpu(), gpurel.gpu());

  BOOST_CHECK_SMALL(cpurel(0)-gpurel(0), TOL);
  BOOST_CHECK_GT(cpurel(0), 1.0);
  BOOST_CHECK_LT(cpurel(0), 10.0);

}

static float sum_g2(MGrid2f& a) {
  float ret = 0;
  for(unsigned i = 0, n = a.dimension(0); i < n; i++) {
    for (unsigned j = 0; j < a.dimension(1); j++) {
      ret += a(i,j);
    }
  }
  return ret;
}

static float sum_g4(MGrid4f& a) {
  a.tocpu();
  float ret = 0;
  for(unsigned i = 0, n = a.dimension(0); i < n; i++) {
    for (unsigned j = 0; j < a.dimension(1); j++) {
      for (unsigned k = 0; k < a.dimension(2); k++) {
        for (unsigned m = 0; m < a.dimension(3); m++) {
          ret += a(i,j,k,m);
        }
      }
    }
  }
  return ret;
}


BOOST_AUTO_TEST_CASE(backward_gradients) {
  //make sure cpu matches gpu
  using namespace std;
  GridMaker g(0.1, 6.0);

  vector<float3> c { make_float3(0, 0, 0) };
  vector<vector<float> > t { vector<float>{0,1.0} };
  vector<float> r { 2.0 };

  CoordinateSet coords(c, t, r);
  float dim = g.get_grid_dims().x;
  MGrid4f diff(2, dim, dim, dim);
  diff(0, 32, 34, 36) = 1.0;
  diff(1, 32, 34, 36) = -1.0;

  MGrid2f cpuatoms(1, 3);
  MGrid2f cputypes(1, 2);
  g.backward(float3 { 0, 0, 0 }, coords, diff.cpu(), cpuatoms.cpu(), cputypes.cpu());

  MGrid4f diffdiff(2, dim,dim,dim);
  MGrid2f add(1,3);
  MGrid2f tdd(1,2);

  g.backward_gradients(float3{0,0,0}, coords, diff.cpu(), cpuatoms.cpu(), cputypes.cpu(), diffdiff.cpu(), add.cpu(), tdd.cpu());

  MGrid4f gpu_diffdiff(2, dim,dim,dim);
  MGrid2f gpu_add(1,3);
  MGrid2f gpu_tdd(1,2);

  std::cerr << "DIFF "<< diff(1,30,30,30) << "\n";
  g.backward_gradients(float3{0,0,0}, coords, diff.gpu(), cpuatoms.gpu(), cputypes.gpu(), gpu_diffdiff.gpu(), gpu_add.gpu(), gpu_tdd.gpu());

  same_grids(gpu_diffdiff,diffdiff);
  same_coords(add,gpu_add);
  BOOST_CHECK_EQUAL(sum_g2(tdd),0);
  BOOST_CHECK_EQUAL(sum_g2(gpu_tdd),0);
  BOOST_CHECK_NE(sum_g2(add),0);
  BOOST_CHECK_NE(sum_g4(diffdiff),0);

}

BOOST_AUTO_TEST_CASE(backward_vec) {
  using namespace std;
  GridMaker g(0.1, 6.0);

  vector<float3> c { make_float3(0, 0, 0) };
  vector<vector<float> > t { vector<float>{0,1.0} };
  vector<float> r { 2.0 };

  CoordinateSet coords(c, t, r);
  float dim = g.get_grid_dims().x;
  MGrid4f diff(2, dim, dim, dim);
  diff(0, 30, 30, 30) = 1.0;

  MGrid2f cpuatoms(1, 3);
  MGrid2f cputypes(1, 2);
  g.backward(float3 { 0, 0, 0 }, coords, diff.cpu(), cpuatoms.cpu(), cputypes.cpu());

  BOOST_CHECK_GT(cputypes[0][0],0);
  BOOST_CHECK_EQUAL(cputypes[0][1],0);
}

BOOST_AUTO_TEST_CASE(forward_vec) {
  using namespace std;
  GridMaker g(0.1, 6.0);

  vector<float3> c { make_float3(0, 0, 0) };
  vector<vector<float> > t { vector<float>{0.5,0.5} };
  vector<float> r { 2.0 };

  CoordinateSet coords(c, t, r);
  float dim = g.get_grid_dims().x;
  MGrid4f out(2, dim, dim, dim);
  MGrid4f out2(2, dim, dim, dim);

  g.forward(float3{0,0,0}, coords, out.cpu());
  g.forward(float3{0,0,0}, coords, out2.gpu());

  unsigned sz = out.size();
  float *g1 = out.data();
  float *g2 = out2.data();
  float max = 0;
  for(unsigned i = 0; i < sz; i++) {
    BOOST_CHECK_SMALL(g1[i] - g2[i], TOL);
    if(g1[i] > max) max = g1[i];
  }
  BOOST_CHECK_EQUAL(max, .5); //make sure not zero
}

//test using radii from types
BOOST_AUTO_TEST_CASE(forward_vec_types) {
  using namespace std;
  GridMaker ga(0.5, 6.0);
  GridMaker gt(0.5, 6.0, false, true);

  vector<float3> ct { make_float3(0, 0, 0) };
  vector<vector<float> > tt { vector<float>{1.0,1.0,1.0} };
  vector<float> rt { 3.0, 2.0, 1.0 };

  vector<float3> ca { make_float3(0, 0, 0), make_float3(0,0,0), make_float3(0,0,0) };
  vector<float> ra { 3.0, 2.0, 1.0 };
  vector<vector<float> > ta { vector<float>{1.0,0.0,0.0}, vector<float>{0.0,1.0,0.0}, vector<float>{0.0,0.0,1.0} };

  CoordinateSet coordsa(ca, ta, ra);
  CoordinateSet coordst(ct, tt, rt);

  float dim = ga.get_grid_dims().x;
  MGrid4f outa(3, dim, dim, dim);
  MGrid4f outt(3, dim, dim, dim);

  ga.forward(float3{0,0,0}, coordsa, outa.cpu());
  gt.forward(float3{0,0,0}, coordst, outt.cpu());

  BOOST_CHECK_EQUAL(gt.get_radii_type_indexed(),true);
  BOOST_CHECK_EQUAL(ga.get_radii_type_indexed(),false);

  unsigned sz = outa.size();
  float *Ga = outa.data();
  float *Gt = outt.data();
  float max = 0;
  for(unsigned i = 0; i < sz; i++) {
    BOOST_CHECK_SMALL(Ga[i] - Gt[i], TOL);
    if(Ga[i] > max) max = Ga[i];
  }

  //check gpu code
  outt.fill_zero();
  gt.forward(float3{0,0,0}, coordst, outt.gpu());
  Gt = outt.data(); //bring back to cpu
  for(unsigned i = 0; i < sz; i++) {
    BOOST_CHECK_SMALL(Ga[i] - Gt[i], TOL);
  }

  BOOST_CHECK_EQUAL(max, 1.0); //make sure not zero

  //check that types go in right channel
  vector<float> ra2 { 2.0, 2.0, 2.0 };
  CoordinateSet coords(ca,ta, ra2);
  gt.forward(float3{0,0,0}, coords, outt.cpu());
  for(unsigned i = 0; i < dim; i++)
    for(unsigned j = 0; j < dim; j++)
      for(unsigned k = 0; k < dim; k++) {
        BOOST_CHECK_SMALL(outt(0,i,j,k)-outt(1,i,j,k),TOL);
        BOOST_CHECK_SMALL(outt(1,i,j,k)-outt(2,i,j,k),TOL);
      }


}

