/*
 * grid_maker.cpp
 *
 *  Created on: Mar 29, 2019
 *      Author: dkoes
 */
#include "libmolgrid/grid_maker.h"
#include <cmath>
#include <vector>

namespace libmolgrid {


void GridMaker::initialize(float res, float d, bool bin, float rscale, float grm) {
  resolution = res;
  dimension = d;
  radius_scale = rscale;
  gaussian_radius_multiple = grm;
  final_radius_multiple = (1+2*grm*grm)/(2*grm);
  dim = ::round(dimension / resolution) + 1;
  binary = bin;

  A = exp(-2*grm*grm)*4*grm*grm; // *d^2/r^2
  B = -exp(-2*grm*grm)*(4*grm+8*grm*grm*grm); // * d/r
  C = exp(-2*grm*grm)*(4*grm*grm*grm*grm+4*grm*grm+1); //constant

  D = 8*grm*grm*exp(-2.0*grm*grm); // * d/r^2
  E = - ( 4*grm + 8*grm*grm*grm) * exp(-2*grm*grm); // * 1/r
}

//validate argument ranges
template<typename Dtype, bool isCUDA>
void GridMaker::check_index_args(const Grid<float, 2, isCUDA>& coords,
    const Grid<float, 1, isCUDA>& type_index, const Grid<float, 1, isCUDA>& radii,
    Grid<Dtype, 4, isCUDA>& out) const {

  size_t N = coords.dimension(0);

  if(dim != out.dimension(1)) throw std::out_of_range("Output grid dimension incorrect: "+itoa(dim) +" vs " +itoa(out.dimension(1)));
  if(dim != out.dimension(2)) throw std::out_of_range("Output grid dimension incorrect: "+itoa(dim) +" vs " +itoa(out.dimension(2)));
  if(dim != out.dimension(3)) throw std::out_of_range("Output grid dimension incorrect: "+itoa(dim) +" vs " +itoa(out.dimension(3)));

  if(type_index.size() != N) throw std::out_of_range("type_index does not match number of atoms: "+itoa(type_index.size())+" vs "+itoa(N));
  if(radii.size() != N) throw std::out_of_range("radii does not match number of atoms: "+itoa(radii.size())+" vs "+itoa(N));

}

template void GridMaker::check_index_args(const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    Grid<float, 4, false>& out) const;
template void GridMaker::check_index_args(const Grid<float, 2, true>& coords,
    const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
    Grid<float, 4, true>& out) const;
template void GridMaker::check_index_args(const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    Grid<double, 4, false>& out) const;
template void GridMaker::check_index_args(const Grid<float, 2, true>& coords,
    const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
    Grid<double, 4, true>& out) const;

float3 GridMaker::get_grid_origin(const float3& grid_center) const {
  float half = dimension / 2.0;
  float3 grid_origin;
  grid_origin.x = grid_center.x - half;
  grid_origin.y = grid_center.y - half;
  grid_origin.z = grid_center.z - half;
  return grid_origin;
}

template <typename Dtype, bool isCUDA>
void GridMaker::forward(const Example& in, const Transform& transform, Grid<Dtype, 4, isCUDA>& out) const {
  CoordinateSet c = in.merge_coordinates(); // !important - this copies the underlying coordinates so we can safely mogrify them
  if(c.max_type != out.dimension(0)) throw std::out_of_range("Incorrect number of channels in output grid: "+itoa(c.max_type) +" vs "+itoa(out.dimension(0)));
  if(isCUDA) c.togpu(); //this will enable tranformation on the gpu
  transform.forward(c,c);
  forward(transform.get_rotation_center(), c, out);
}

//not sure why these have to be instantiated given the next function must implicitly isntantiate them
template void GridMaker::forward(const Example& in, const Transform& transform, Grid<float, 4, false>& out) const;
template void GridMaker::forward(const Example& in, const Transform& transform, Grid<float, 4, true>& out) const;
template void GridMaker::forward(const Example& in, const Transform& transform, Grid<double, 4, false>& out) const;
template void GridMaker::forward(const Example& in, const Transform& transform, Grid<double, 4, true>& out) const;


template<typename Dtype, bool isCUDA>
void GridMaker::forward(const Example& in, Grid<Dtype, 4, isCUDA>& out,
    float random_translation, bool random_rotation, const float3& center) const {
  float3 c = center;
  if(std::isinf(c.x)) {
    c = in.sets.back().center();
  }
  Transform t(c, random_translation, random_rotation);
  forward(in, t, out);
}

template void GridMaker::forward(const Example& in, Grid<float, 4, false>& out,
    float random_translation, bool random_rotation, const float3& center) const;
template void GridMaker::forward(const Example& in, Grid<float, 4, true>& out,
    float random_translation, bool random_rotation, const float3& center) const;
template void GridMaker::forward(const Example& in, Grid<double, 4, false>& out,
    float random_translation, bool random_rotation, const float3& center) const;
template void GridMaker::forward(const Example& in, Grid<double, 4, true>& out,
    float random_translation, bool random_rotation, const float3& center) const;

template<typename Dtype>
void GridMaker::forward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    Grid<Dtype, 4, false>& out) const {
  //zero grid first
  std::fill(out.data(), out.data() + out.size(), 0.0);
  check_index_args(coords, type_index, radii, out);

  float3 grid_origin = get_grid_origin(grid_center);
  size_t natoms = coords.dimension(0);
  size_t ntypes = out.dimension(0);
  //iterate over all atoms
  for (size_t aidx = 0; aidx < natoms; ++aidx) {
    float atype = type_index(aidx);
    if(atype >= ntypes) throw std::out_of_range("Type index "+itoa(atype)+" larger than allowed "+itoa(ntypes));
    if (atype >= 0 && atype < ntypes) {
      float3 acoords;
      acoords.x = coords(aidx, 0);
      acoords.y = coords(aidx, 1);
      acoords.z = coords(aidx, 2);
      float radius = radii(aidx);
      float densityrad = radius * radius_scale * final_radius_multiple;

      uint2 bounds[3];
      bounds[0] = get_bounds_1d(grid_origin.x, coords(aidx, 0), densityrad);
      bounds[1] = get_bounds_1d(grid_origin.y, coords(aidx, 1), densityrad);
      bounds[2] = get_bounds_1d(grid_origin.z, coords(aidx, 2), densityrad);

      //for every grid point possibly overlapped by this atom
      for (size_t i = bounds[0].x, iend = bounds[0].y; i < iend; i++) {
        for (size_t j = bounds[1].x, jend = bounds[1].y; j < jend; j++) {
          for (size_t k = bounds[2].x, kend = bounds[2].y; k < kend; k++) {
            float3 grid_coords;
            grid_coords.x = grid_origin.x + i * resolution;
            grid_coords.y = grid_origin.y + j * resolution;
            grid_coords.z = grid_origin.z + k * resolution;
            float val = calc_point(acoords.x, acoords.y, acoords.z, radius, grid_coords);
            size_t offset = ((((atype * dim) + i) * dim) + j) * dim + k;
            // std::cout << "val " << val << "\n";

            if (binary) {
              if (val != 0)
                *(out.data() + offset) = 1.0;
            }
            else {
              *(out.data() + offset) += val;
            }

          }
        }
      }
    }
  }
}

template void GridMaker::forward(const std::vector<Example>& in, Grid<float, 5, false>& out,
  float random_translation, bool random_rotation) const;
template void GridMaker::forward(const std::vector<Example>& in, Grid<float, 5, true>& out,
    float random_translation, bool random_rotation) const;
template void GridMaker::forward(const std::vector<Example>& in, Grid<double, 5, false>& out,
  float random_translation, bool random_rotation) const;
template void GridMaker::forward(const std::vector<Example>& in, Grid<double, 5, true>& out,
    float random_translation, bool random_rotation) const;

template void GridMaker::forward(float3 grid_center,
    const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    Grid<float, 4, false>& out) const;
template void GridMaker::forward(float3 grid_center,
    const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    Grid<double, 4, false>& out) const;


//set a single atom gradient - note can't pass a slice by reference
template <typename Dtype>
float3 GridMaker::calc_atom_gradient_cpu(const float3& grid_origin, const Grid1f& coord, const Grid<Dtype, 3, false>& diff, float radius) const {

  float3 agrad{0,0,0};

  float r = radius * radius_scale * final_radius_multiple;
  std::array<uint2, 3> ranges;
  ranges[0] = get_bounds_1d(grid_origin.x, coord(0), r);
  ranges[1] = get_bounds_1d(grid_origin.y, coord(1), r);
  ranges[2] = get_bounds_1d(grid_origin.z, coord(2), r);

  float3 a{coord(0),coord(1),coord(2)}; //atom coordinate

  //for every grid point possibly overlapped by this atom
  for (unsigned i = ranges[0].x, iend = ranges[0].y; i < iend;
      ++i) {
    for (unsigned j = ranges[1].x, jend = ranges[1].y; j < jend; ++j) {
      for (unsigned k = ranges[2].x, kend = ranges[2].y; k < kend; ++k) {
        //convert grid point coordinates to angstroms
        float x = grid_origin.x + i * resolution;
        float y = grid_origin.y + j * resolution;
        float z = grid_origin.z + k * resolution;

        accumulate_atom_gradient(a.x,a.y,a.z, x,y,z, radius, diff(i,j,k), agrad);
      }
    }
  }

  return agrad;
}

//cpu backwards
template <typename Dtype>
void GridMaker::backward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    const Grid<Dtype, 4, false>& diff, Grid<Dtype, 2, false>& atom_gradients) const {

  atom_gradients.fill_zero();
  unsigned n = coords.dimension(0);
  if(n != type_index.size()) throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
  if(n != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");

  float3 grid_origin = get_grid_origin(grid_center);

  for (unsigned i = 0; i < n; ++i) {
    int whichgrid = round(type_index[i]); // this is which atom-type channel of the grid to look at
    if (whichgrid >= 0) {
      float3 agrad = calc_atom_gradient_cpu(grid_origin, coords[i], diff[whichgrid], radii[i]);
      atom_gradients(i,0) = agrad.x;
      atom_gradients(i,1) = agrad.y;
      atom_gradients(i,2) = agrad.z;
    }
  }

}

template void GridMaker::backward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    const Grid<float, 4, false>& diff, Grid<float, 2, false>& atom_gradients) const;
template void GridMaker::backward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    const Grid<double, 4, false>& diff, Grid<double, 2, false>& atom_gradients) const;

}
