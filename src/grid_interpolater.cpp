/*
 * grid_interpolater.cpp
 *
 *  Created on: May 5, 2021
 *      Author: dkoes
 */

#include "libmolgrid/grid_interpolater.h"
#include <cmath>
#include <vector>
#include <iomanip>

namespace libmolgrid {


  template <typename Dtype>
  void GridInterpolater::forward(float3 in_center, const Grid<Dtype, 4, false>& in, const Transform& transform, float3 out_center, Grid<Dtype, 4, false>& out) const {

      checkGrids(in, out);
      float3 center = transform.get_rotation_center();
      float in_radius = in_dimension/2.0;
      float out_radius = out_dimension/2.0;
      float3 in_origin = {in_center.x-in_radius,in_center.y-in_radius,in_center.z-in_radius};
      float3 out_origin = {out_center.x-out_radius,out_center.y-out_radius,out_center.z-out_radius};

      Quaternion invQ = transform.get_quaternion().inverse();
      float3 t = transform.get_translation();
      float3 untranslate = {-t.x-center.x, -t.y-center.y, -t.z-center.z};
      unsigned K = in.dimension(0);

      //for every grid point in out
      for(unsigned i = 0; i < out_dim; i++) {
          for(unsigned j = 0; j < out_dim; j++) {
              for(unsigned k = 0; k < out_dim; k++) {
                  //compute its Cartesian location
                  float3 outpt = grid2cart(out_origin, out_resolution, i, j, k);
                  //apply inverse transformation
                  float3 newpt = invQ.rotate(outpt.x+untranslate.x, outpt.y+untranslate.y, outpt.z+untranslate.z);
                  //get (not rounded) input grid coordinates (not Cartesian)
                  float3 inpt = cart2grid(in_origin, in_resolution, newpt.x+center.x, newpt.y+center.y, newpt.z+center.z);
                  //interpolate for each channel
                  for(unsigned c = 0; c < K; c++) {
                      out(c,i,j,k) = interpolate(in[c], inpt);
                  }
              }
          }
      }
  }

  template void GridInterpolater::forward(float3 in_center, const Grid<float, 4, false>& in, const Transform& transform, float3 out_center, Grid<float, 4, false>& out) const;
  template void GridInterpolater::forward(float3 in_center, const Grid<double, 4, false>& in, const Transform& transform, float3 out_center, Grid<double, 4, false>& out) const;


  // Sanity check grid dimensions and throw exceptions if they are wrong
  template <typename Dtype, bool isCUDA>
  void GridInterpolater::checkGrids(const Grid<Dtype, 4, isCUDA>& in, const Grid<Dtype, 4, isCUDA>& out) const {

    if(in.data() == out.data()) {
        throw std::invalid_argument("Input and output grid can not be the same in GridInterpolator.");
    }
    if(in.dimension(0) != out.dimension(0)) {
      throw std::invalid_argument("Different channel dimensions for input and output  grids.");
    }
    if(in.dimension(1) != in_dim || in.dimension(2) != in_dim || in.dimension(3) != in_dim) {
      throw std::invalid_argument("Input grid wrong dimension. Should be "+itoa(in_dim)+" along all axes but is "+itoa(in.dimension(1))+","+itoa(in.dimension(2))+","+itoa(in.dimension(3)));
    }
    if(out.dimension(1) != out_dim || out.dimension(2) != out_dim || out.dimension(3) != out_dim) {
      throw std::invalid_argument("Output grid wrong dimension. Should be "+itoa(out_dim)+" along all axes but is "+itoa(out.dimension(1))+","+itoa(out.dimension(2))+","+itoa(out.dimension(3)));
    }
    if(in.data() == nullptr) {
      throw std::invalid_argument("Input grid not allocated");
    }
    if(out.data() == nullptr) {
      throw std::invalid_argument("Output grid not allocated");
    }
  }

  template void GridInterpolater::checkGrids(const Grid<float, 4, false>& in, const Grid<float, 4, false>& out) const;
  template void GridInterpolater::checkGrids(const Grid<double, 4, false>& in, const Grid<double, 4, false>& out) const;
  template void GridInterpolater::checkGrids(const Grid<float, 4, true>& in, const Grid<float, 4, true>& out) const;

}
