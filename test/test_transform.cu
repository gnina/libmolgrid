/*
 * test_transform.cu
 *
 *  Created on: Jan 10, 2019
 *      Author: dkoes
 */


#define BOOST_TEST_MODULE transform_cuda_test
#include <boost/test/unit_test.hpp>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "libmolgrid/transform.h"
#include "libmolgrid/managed_grid.h"
#include "test_transform.h"

using namespace libmolgrid;


BOOST_AUTO_TEST_CASE(apply_transform)
{
  //non-random transform
  Quaternion q(sqrt(0.5),0,0,sqrt(0.5)); //z 90
  Transform nr(q, make_float3(0,1,1), make_float3(2,0,-3));

  //random
  Transform r(make_float3(0,1,1), 10.0, true);

  float coord_data[8][3] = { {0,0,0},
                         {1,0,0},
                         {0,1,0},
                         {0,0,1},
                         {-1,.5,3},
                         {1,1,1},
                         {0,1,1},
                         {.333,.75,-9}
  };
  float buffer[8][3] = {0,};

  MGrid2f coords(8, 3);

  for(unsigned i = 0; i < 8; i++) {
    for(unsigned j = 0; j < 3; j++) {
      coords[i][j] = coord_data[i][j];
    }
  }
  MGrid2f coords2(8, 3);

  //does nr perform as expected?
  nr.forward(coords.gpu(),coords2.gpu());
  float3 expected = make_float3(2,1,-2);
  eqPt(coords2[6],expected); //at center

  expected = make_float3(2,1,-3);
  eqPt(coords2[2],expected); //on z-axis

  expected = make_float3(2,2,-2);
  eqPt(coords2[5],expected);

  //make sure input unchanged
  expected = make_float3(0.333,.75,-9);
  eqPt(coords[7],expected);

  //does random work both ways
  r.forward(coords.gpu(),coords2.gpu());
  for(unsigned i = 0; i < 8; i++) {
    neqPt(coords[i],coords2[i]);
  }

  //cpu calculation
  Grid2f cpucoords(coords.cpu().data(),8,3);
  Grid2f cpucoords2((float*)buffer,8,3);
  r.forward(cpucoords,cpucoords2);
  for(unsigned i = 0; i < 8; i++) {
    eqPt(cpucoords2[i],coords2[i]);
  }

  r.backward(coords2.gpu(),coords2.gpu());
  coords2.tocpu(); //not pleased that we have to do this..

  for(unsigned i = 0; i < 8; i++) {
    std::cerr << "(" << coords[i][0]<<","<<coords[i][1]<<","<<coords[i][2]<<")  ("<<coords2[i][0]<<","<<coords2[i][1]<<","<<coords2[i][2]<<")  ("<<cpucoords2[i][0]<<","<<cpucoords2[i][1]<<","<<cpucoords2[i][2]<<")\n";
    eqPt(coords[i],coords2[i]);
  }
}
