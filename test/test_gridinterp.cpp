#define BOOST_TEST_MODULE gridinterp_test
#include <boost/test/unit_test.hpp>
#include "test_util.h"
#include "libmolgrid/grid_maker.h"
#include "libmolgrid/grid_interpolater.h"
#include "libmolgrid/example_extractor.h"
#include <iostream>
#include <iomanip>

#define TOL 0.0001f
using namespace libmolgrid;

BOOST_AUTO_TEST_CASE(forward_cpu) {

  MGrid4f src(1,5,5,5);
  MGrid4f dst(1,3,3,3);
  GridInterpolater gi(0.5,2.0,1.0,2.0);
  Transform t;

  float *ptr = src.data();
  for(unsigned i = 0, n = src.size(); i < n; i++) {
      *ptr = float(i);
  }

  gi.forward(src.cpu(),t,dst.cpu());

  for(unsigned i = 0; i < 3; i++) {
      for(unsigned j = 0; j < 3; j++) {
          for(unsigned k = 0; k < 3; k++) {
              BOOST_CHECK_EQUAL(dst[0][i][j][k], src[0][i*2][j*2][k*2]);
          }
      }
  }

}


BOOST_AUTO_TEST_CASE(forward_gpu) {

  MGrid4f src(1,5,5,5);
  MGrid4f dst(1,3,3,3);
  GridInterpolater gi(0.5,2.0,1.0,2.0);
  Transform t;

  float *ptr = src.data();
  for(unsigned i = 0, n = src.size(); i < n; i++) {
      *ptr = float(i);
  }

  gi.forward(src.gpu(),t,dst.gpu());

  for(unsigned i = 0; i < 3; i++) {
      for(unsigned j = 0; j < 3; j++) {
          for(unsigned k = 0; k < 3; k++) {
              BOOST_CHECK_EQUAL(dst[0][i][j][k], src[0][i*2][j*2][k*2]);
          }
      }
  }

}
