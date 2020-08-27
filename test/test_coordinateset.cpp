#define BOOST_TEST_MODULE coordinate_test
#include <boost/test/unit_test.hpp>
#include "test_util.h"
#include "libmolgrid/coordinateset.h"
#include "libmolgrid/transform.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <openbabel/obconversion.h>
#include <openbabel/mol.h>

#define TOL 0.0001f
using namespace libmolgrid;
using namespace std;

BOOST_AUTO_TEST_CASE(coordinates) {
  // hard-coded example, compared with a reference
  // read in example
  vector<float3> coords{make_float3(1,0,-1),make_float3(1,3,-1),make_float3(1,0,-1)};
  vector<int> types{3,2,1};
  vector<float> radii{1.5,1.5,1.0};
  CoordinateSet c(coords,types,radii, 4);
  Transform t(Quaternion(), {0,0,0}, {-1,0,1});
  t.forward(c,c);
  BOOST_CHECK_SMALL(c.coords[1][1]-3.0f,TOL);
  BOOST_CHECK_SMALL(c.coords[0][0],TOL);
    
  CoordinateSet c2 = c.clone();
  c2.coords(1,1) = 0;
  BOOST_CHECK_SMALL(c.coords(1,1)-3.0f,TOL);

}

BOOST_AUTO_TEST_CASE(vectortyper) {
    using namespace OpenBabel;
    OBConversion conv("../../test/data/structs/1apv/1apv_rec.pdb");
    OBMol mol;
    conv.Read(&mol);
    GninaVectorTyper typer;
    CoordinateSet c(&mol, typer);

    BOOST_CHECK(c.has_vector_types());

    BOOST_CHECK_EQUAL(c.type_vector.dimension(0),4546);
    BOOST_CHECK_EQUAL(c.type_vector.dimension(1),26);

    double maxval = 0;
    auto& vec = c.type_vector;
    for(unsigned i = 0; i < vec.dimension(0); i++) {
        for(unsigned j = 0; j < vec.dimension(1); j++) {
            if(vec(i,j) > maxval) maxval = vec(i,j);
        }
    }
    BOOST_CHECK_LT(maxval, 34);

}
