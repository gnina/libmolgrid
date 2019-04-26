/*
 * grid_maker.cpp
 *
 *  Created on: Mar 29, 2019
 *      Author: dkoes
 */
#include "libmolgrid/grid_maker.h"
#include <cmath>

namespace libmolgrid {

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

std::pair<size_t, size_t> GridMaker::get_bounds_1d(const float grid_origin,
    float coord,
    float densityrad) const {
  std::pair<size_t, size_t> bounds(0, 0);
  float low = coord - densityrad - grid_origin;
  if (low > 0) {
    bounds.first = floor(low / resolution);
  }

  float high = coord + densityrad - grid_origin;
  if (high > 0) //otherwise zero
      {
    bounds.second = std::min(dim, (size_t) ceil(high / resolution));
  }
  return bounds;
}

template <typename Dtype, bool isCUDA>
void GridMaker::forward(const Example& in, const Transform& transform, Grid<Dtype, 4, isCUDA>& out) const {
  CoordinateSet c = in.merge_coordinates(); // !important - this copies the underlying coordinates so we can safely mogrify them
  if(c.max_type != out.dimension(0)) throw std::out_of_range("Incorrect number of channels in output grid: "+itoa(c.max_type) +" vs "+itoa(out.dimension(0)));
  if(isCUDA) c.togpu(); //this will enable tranformation on the gpu
  transform.forward(c,c);
  forward(transform.rotation_center(), c, out);
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
      float densityrad = radius * radiusmultiple;

      std::array<std::pair<size_t, size_t>, 3> bounds;
      bounds[0] = get_bounds_1d(grid_origin.x, coords(aidx, 0), densityrad);
      bounds[1] = get_bounds_1d(grid_origin.y, coords(aidx, 1), densityrad);
      bounds[2] = get_bounds_1d(grid_origin.z, coords(aidx, 2), densityrad);
      // std::cout << "coords.x " << acoords.x;
      // std::cout << " coords.y " << acoords.y;
      // std::cout << " coords.z " << acoords.z << "\n";
      // std::cout << "bounds[0].first " << bounds[0].first;
      // std::cout << " bounds[0].second " << bounds[0].second;
      // std::cout << " bounds[1].first " << bounds[1].first;
      // std::cout << " bounds[1].second " << bounds[1].second;
      // std::cout << " bounds[2].first " << bounds[2].first;
      // std::cout << " bounds[2].second " << bounds[2].second << "\n";

      //for every grid point possibly overlapped by this atom
      for (size_t i = bounds[0].first, iend = bounds[0].second; i < iend; i++) {
        for (size_t j = bounds[1].first, jend = bounds[1].second; j < jend;
            j++) {
          for (size_t k = bounds[2].first, kend = bounds[2].second;
              k < kend; k++) {
            float3 grid_coords;
            grid_coords.x = grid_origin.x + i * resolution;
            grid_coords.y = grid_origin.y + j * resolution;
            grid_coords.z = grid_origin.z + k * resolution;
            float val = calc_point(acoords, radius, grid_coords);
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

}
