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
  void GridInterpolater::forward(Vec3 in_center, const Grid<Dtype, 4, false>& in, const Transform& transform, Vec3 out_center, Grid<Dtype, 4, false>& out) const {

      checkGrids(in, out);
      Vec3 center = transform.get_rotation_center();
      float in_radius = in_dimension/2.0;
      float out_radius = out_dimension/2.0;
      Vec3 in_origin = {in_center.x-in_radius,in_center.y-in_radius,in_center.z-in_radius};
      Vec3 out_origin = {out_center.x-out_radius,out_center.y-out_radius,out_center.z-out_radius};

      Quaternion invQ = transform.get_quaternion().inverse();
      Vec3 t = transform.get_translation();
      Vec3 untranslate = {-t.x-center.x, -t.y-center.y, -t.z-center.z};
      unsigned K = in.dimension(0);

      //for every grid point in out
      for(unsigned i = 0; i < out_dim; i++) {
          for(unsigned j = 0; j < out_dim; j++) {
              for(unsigned k = 0; k < out_dim; k++) {
                  //compute its Cartesian location
                  Vec3 outpt = grid2cart(out_origin, out_resolution, i, j, k);
                  //apply inverse transformation
                  Vec3 newpt = invQ.rotate(outpt.x+untranslate.x, outpt.y+untranslate.y, outpt.z+untranslate.z);
                  //get (not rounded) input grid coordinates (not Cartesian)
                  Vec3 inpt = cart2grid(in_origin, in_resolution, newpt.x+center.x, newpt.y+center.y, newpt.z+center.z);
                  //interpolate for each channel
                  for(unsigned c = 0; c < K; c++) {
                      out(c,i,j,k) = interpolate(in[c], inpt);
                  }
              }
          }
      }
  }

  template void GridInterpolater::forward(Vec3 in_center, const Grid<float, 4, false>& in, const Transform& transform, Vec3 out_center, Grid<float, 4, false>& out) const;
  template void GridInterpolater::forward(Vec3 in_center, const Grid<double, 4, false>& in, const Transform& transform, Vec3 out_center, Grid<double, 4, false>& out) const;


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

#if !LIBMOLGRID_USE_CUDA
  // ---------------------------------------------------------------------------
  // Definitions previously in grid_interpolater.cu (CPU-callable)
  // ---------------------------------------------------------------------------

  template <typename Dtype, bool isCUDA>
  Dtype GridInterpolater::get_pt(const Grid<Dtype, 3, isCUDA>& in, int x, int y, int z) const {
    if (x < 0 || x >= int(in_dim) || y < 0 || y >= int(in_dim) || z < 0 || z >= int(in_dim))
      return 0;
    return in[x][y][z];
  }

  template <typename Dtype, bool isCUDA>
  Dtype GridInterpolater::interpolate(const Grid<Dtype, 3, isCUDA>& in, Vec3 gridpt) const {
    int xl = (int)floor(gridpt.x), xh = (int)ceil(gridpt.x);
    int yl = (int)floor(gridpt.y), yh = (int)ceil(gridpt.y);
    int zl = (int)floor(gridpt.z), zh = (int)ceil(gridpt.z);

    Dtype p000 = get_pt(in,xl,yl,zl), p001 = get_pt(in,xl,yl,zh);
    Dtype p010 = get_pt(in,xl,yh,zl), p011 = get_pt(in,xl,yh,zh);
    Dtype p100 = get_pt(in,xh,yl,zl), p101 = get_pt(in,xh,yl,zh);
    Dtype p110 = get_pt(in,xh,yh,zl), p111 = get_pt(in,xh,yh,zh);

    Dtype xd = xh > xl ? (gridpt.x-xl)/(xh-xl) : 0;
    Dtype yd = yh > yl ? (gridpt.y-yl)/(yh-yl) : 0;
    Dtype zd = zh > zl ? (gridpt.z-zl)/(zh-zl) : 0;

    Dtype c00 = p000*(1-xd) + p100*xd;
    Dtype c01 = p001*(1-xd) + p101*xd;
    Dtype c10 = p010*(1-xd) + p110*xd;
    Dtype c11 = p011*(1-xd) + p111*xd;
    Dtype c0  = c00*(1-yd)  + c10*yd;
    Dtype c1  = c01*(1-yd)  + c11*yd;
    return c0*(1-zd) + c1*zd;
  }

  template float  GridInterpolater::interpolate(const Grid<float,  3, true>&  in, Vec3 pt) const;
  template float  GridInterpolater::interpolate(const Grid<float,  3, false>& in, Vec3 pt) const;
  template double GridInterpolater::interpolate(const Grid<double, 3, false>& in, Vec3 pt) const;
#endif

}
