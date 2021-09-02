/*
 * grid_maker.cpp
 *
 *  Created on: Mar 29, 2019
 *      Author: dkoes
 */
#include "libmolgrid/grid_maker.h"
#include <cmath>
#include <vector>
#include <iomanip>

namespace libmolgrid {


void GridMaker::initialize(float res, float d, bool bin, float rscale, float grm) {
  resolution = res;
  dimension = d;
  radius_scale = rscale;
  gaussian_radius_multiple = grm;
  if(gaussian_radius_multiple < 0) {
    //try to interpret this is requesting a truncated gradient - no quadratic term
    gaussian_radius_multiple *= -1;
    final_radius_multiple = gaussian_radius_multiple;
  } else {
    final_radius_multiple = (1+2*grm*grm)/(2*grm);
  }
  dim = std::round(dimension / resolution) + 1;
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
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii, Grid<float, 4, false>& out) const;
template void GridMaker::check_index_args(const Grid<float, 2, true>& coords,
    const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii, Grid<float, 4, true>& out) const;
template void GridMaker::check_index_args(const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii, Grid<double, 4, false>& out) const;
template void GridMaker::check_index_args(const Grid<float, 2, true>& coords,
    const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii, Grid<double, 4, true>& out) const;


//validate argument ranges
template<typename Dtype, bool isCUDA>
void GridMaker::check_vector_args(const Grid<float, 2, isCUDA>& coords,
    const Grid<float, 2, isCUDA>& type_vector, const Grid<float, 1, isCUDA>& radii,
    const Grid<Dtype, 4, isCUDA>& out) const {

  size_t N = coords.dimension(0);
  size_t T = type_vector.dimension(1);

  if(dim != out.dimension(1)) throw std::out_of_range("Output grid dimension incorrect: "+itoa(dim) +" vs " +itoa(out.dimension(1)));
  if(dim != out.dimension(2)) throw std::out_of_range("Output grid dimension incorrect: "+itoa(dim) +" vs " +itoa(out.dimension(2)));
  if(dim != out.dimension(3)) throw std::out_of_range("Output grid dimension incorrect: "+itoa(dim) +" vs " +itoa(out.dimension(3)));

  if(type_vector.dimension(0) != N)
    throw std::out_of_range("type_vector does not match number of atoms: "+itoa(type_vector.dimension(0))+" vs "+itoa(N));
  if(T != out.dimension(0))
    throw std::out_of_range("number of types in type_vector does not match number of output channels: "+itoa(T)+" vs "+itoa(out.dimension(0)));

  if(radii_type_indexed) {
    if(radii.size() != T) throw std::out_of_range("radii does not match number of types: "+itoa(radii.size())+" vs "+itoa(T));
  } else {
    if(radii.size() != N) throw std::out_of_range("radii does not match number of atoms: "+itoa(radii.size())+" vs "+itoa(N));
  }
}

template void GridMaker::check_vector_args(const Grid<float, 2, false>& coords,
    const Grid<float, 2, false>& type_vec, const Grid<float, 1, false>& radii, const Grid<float, 4, false>& out) const;
template void GridMaker::check_vector_args(const Grid<float, 2, true>& coords,
    const Grid<float, 2, true>& type_vec, const Grid<float, 1, true>& radii, const Grid<float, 4, true>& out) const;
template void GridMaker::check_vector_args(const Grid<float, 2, false>& coords,
    const Grid<float, 2, false>& type_vec, const Grid<float, 1, false>& radii, const Grid<double, 4, false>& out) const;
template void GridMaker::check_vector_args(const Grid<float, 2, true>& coords,
    const Grid<float, 2, true>& type_vec, const Grid<float, 1, true>& radii, const Grid<double, 4, true>& out) const;

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

//not sure why these have to be instantiated given the next function must implicitly instantiate them
template void GridMaker::forward(const Example& in, const Transform& transform, Grid<float, 4, false>& out) const;
template void GridMaker::forward(const Example& in, const Transform& transform, Grid<float, 4, true>& out) const;
template void GridMaker::forward(const Example& in, const Transform& transform, Grid<double, 4, false>& out) const;
template void GridMaker::forward(const Example& in, const Transform& transform, Grid<double, 4, true>& out) const;


template<typename Dtype, bool isCUDA>
void GridMaker::forward(const Example& in, Grid<Dtype, 4, isCUDA>& out,
    float random_translation, bool random_rotation, const float3& center) const {
  float3 c = center;
  if(std::isinf(c.x)) {
      for(int i = (int)in.sets.size()-1; i >= 0; i--) {
          //last non-empty coordinate set
          c = float3{0,0,0};
          if(in.sets[i].size()) {
              c = in.sets.back().center();
              break;
          }
      }
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
  if(radii_type_indexed) {
    throw std::invalid_argument("Type indexed radii not supported with index types.");
  }

  float3 grid_origin = get_grid_origin(grid_center);
  size_t natoms = coords.dimension(0);
  size_t ntypes = out.dimension(0);
  Dtype *data = out.data();

  //iterate over all atoms
  for (size_t aidx = 0; aidx < natoms; ++aidx) {
    size_t atype = floor(type_index(aidx));
    if(atype >= ntypes) throw std::out_of_range("Type index "+itoa(atype)+" larger than allowed "+itoa(ntypes));
    if (atype >= 0 && atype < ntypes) {
      float3 acoords;
      acoords.x = coords(aidx, 0);
      acoords.y = coords(aidx, 1);
      acoords.z = coords(aidx, 2);
      float radius = radii(aidx);
      float densityrad = radius * radius_scale * final_radius_multiple;

      uint2 bounds[3];
      bounds[0] = get_bounds_1d(grid_origin.x, acoords.x, densityrad);
      bounds[1] = get_bounds_1d(grid_origin.y, acoords.y, densityrad);
      bounds[2] = get_bounds_1d(grid_origin.z, acoords.z, densityrad);

      //for every grid point possibly overlapped by this atom
      for (size_t i = bounds[0].x, iend = bounds[0].y; i < iend; i++) {
        for (size_t j = bounds[1].x, jend = bounds[1].y; j < jend; j++) {
          for (size_t k = bounds[2].x, kend = bounds[2].y; k < kend; k++) {
            float3 grid_coords;
            grid_coords.x = grid_origin.x + i * resolution;
            grid_coords.y = grid_origin.y + j * resolution;
            grid_coords.z = grid_origin.z + k * resolution;

            size_t offset = ((((atype * dim) + i) * dim) + j) * dim + k;
            if (binary) {
              float val = calc_point<true>(acoords.x, acoords.y, acoords.z, radius, grid_coords);

              if (val != 0)
                *(data + offset) = 1.0;
            }
            else {
              float val = calc_point<false>(acoords.x, acoords.y, acoords.z, radius, grid_coords);
              *(data + offset) += val;
            }

          }
        }
      }
    }
  }
}

template<typename Dtype>
void GridMaker::forward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 2, false>& type_vector, const Grid<float, 1, false>& radii,
    Grid<Dtype, 4, false>& out) const {
  //zero grid first
  std::fill(out.data(), out.data() + out.size(), 0.0);
  check_vector_args(coords, type_vector, radii, out);

  float3 grid_origin = get_grid_origin(grid_center);
  size_t natoms = coords.dimension(0);
  size_t ntypes = type_vector.dimension(1);
  //iterate over all atoms
  for (size_t aidx = 0; aidx < natoms; ++aidx) {
    for (size_t tidx = 0; tidx < ntypes; tidx++) {
      Dtype tmult = type_vector(aidx, tidx); //amount of type for this atom
      if (tmult != 0) {
        float3 acoords;
        acoords.x = coords(aidx, 0);
        acoords.y = coords(aidx, 1);
        acoords.z = coords(aidx, 2);
        float radius = 0;
        if(radii_type_indexed) radius = radii(tidx);
        else radius = radii(aidx);

        float densityrad = radius * radius_scale * final_radius_multiple;

        uint2 bounds[3];
        bounds[0] = get_bounds_1d(grid_origin.x, acoords.x, densityrad);
        bounds[1] = get_bounds_1d(grid_origin.y, acoords.y, densityrad);
        bounds[2] = get_bounds_1d(grid_origin.z, acoords.z, densityrad);

        //for every grid point possibly overlapped by this atom
        for (size_t i = bounds[0].x, iend = bounds[0].y; i < iend; i++) {
          for (size_t j = bounds[1].x, jend = bounds[1].y; j < jend; j++) {
            for (size_t k = bounds[2].x, kend = bounds[2].y; k < kend; k++) {
              float3 grid_coords;
              grid_coords.x = grid_origin.x + i * resolution;
              grid_coords.y = grid_origin.y + j * resolution;
              grid_coords.z = grid_origin.z + k * resolution;

              size_t offset = ((((tidx * dim) + i) * dim) + j) * dim + k;
              if (binary) {
                float val = calc_point<true>(acoords.x, acoords.y, acoords.z, radius, grid_coords);

                if (val != 0)
                  *(out.data() + offset) += tmult; //not quite binary
              }
              else {
                float val = calc_point<false>(acoords.x, acoords.y, acoords.z, radius, grid_coords);
                *(out.data() + offset) += val*tmult;
              }
            }
          }
        }
      } //tmult != 0
    } //tidx
  } //aidx
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
    const Grid<float, 1, false>& type_index, 
    const Grid<float, 1, false>& radii,
    Grid<float, 4, false>& out) const;
template void GridMaker::forward(float3 grid_center,
    const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, 
    const Grid<float, 1, false>& radii,
    Grid<double, 4, false>& out) const;

template void GridMaker::forward(float3 grid_center, const Grid<float, 2, false>& coords,
        const Grid<float, 2, false>& type_vector, const Grid<float, 1, false>& radii,
        Grid<float, 4, false>& out) const;
template void GridMaker::forward(float3 grid_center, const Grid<float, 2, false>& coords,
        const Grid<float, 2, false>& type_vector, const Grid<float, 1, false>& radii,
        Grid<double, 4, false>& out) const;

//batched cpu float

template void GridMaker::forward<float,2,false>(const Grid<float, 2, false> &centers,
    const Grid<float, 3, false> &coords,
    const Grid<float, 2, false> &types,
    const Grid<float, 2, false> &radii, Grid<float, 5, false> &out) const;
template void GridMaker::forward<float,3,false>(const Grid<float, 2, false> &centers,
    const Grid<float, 3, false> &coords,
    const Grid<float, 3, false> &types,
    const Grid<float, 2, false> &radii,Grid<float, 5, false> &out) const;

//batched cpu double
template void GridMaker::forward<double,2,false>(const Grid<float, 2, false> &centers,
    const Grid<float, 3, false> &coords,
    const Grid<float, 2, false> &types,
    const Grid<float, 2, false> &radii, Grid<double, 5, false> &out) const;
template void GridMaker::forward<double,3,false>(const Grid<float, 2, false> &centers,
    const Grid<float, 3, false> &coords,
    const Grid<float, 3, false> &types,
    const Grid<float, 2, false> &radii,Grid<double, 5, false> &out) const;



//set a single atom gradient - note can't pass a slice by reference
template <typename Dtype>
float3 GridMaker::calc_atom_gradient_cpu(const float3& grid_origin, const Grid1f& coordr, const Grid<Dtype, 3, false>& diff, float radius) const {

  float3 agrad{0,0,0};
  float r = radius * radius_scale * final_radius_multiple;
  float3 a{coordr(0),coordr(1),coordr(2)}; //atom coordinate

  uint2 ranges[3];
  ranges[0] = get_bounds_1d(grid_origin.x, a.x, r);
  ranges[1] = get_bounds_1d(grid_origin.y, a.y, r);
  ranges[2] = get_bounds_1d(grid_origin.z, a.z, r);


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

//set return accumulated gradient of type
template <typename Dtype>
float GridMaker::calc_type_gradient_cpu(const float3& grid_origin, const Grid1f& coordr, const Grid<Dtype, 3, false>& diff, float radius) const {

  float r = radius * radius_scale * final_radius_multiple;
  float3 a{coordr(0),coordr(1),coordr(2)}; //atom coordinate

  uint2 ranges[3];
  ranges[0] = get_bounds_1d(grid_origin.x, a.x, r);
  ranges[1] = get_bounds_1d(grid_origin.y, a.y, r);
  ranges[2] = get_bounds_1d(grid_origin.z, a.z, r);

  float ret = 0;
  //for every grid point possibly overlapped by this atom
  for (unsigned i = ranges[0].x, iend = ranges[0].y; i < iend; ++i) {
    for (unsigned j = ranges[1].x, jend = ranges[1].y; j < jend; ++j) {
      for (unsigned k = ranges[2].x, kend = ranges[2].y; k < kend; ++k) {
        //convert grid point coordinates to angstroms
        float x = grid_origin.x + i * resolution;
        float y = grid_origin.y + j * resolution;
        float z = grid_origin.z + k * resolution;
        float val;
        if(binary)
          val = calc_point<true>(a.x, a.y, a.z, radius, float3{x,y,z});
        else
          val = calc_point<false>(a.x, a.y, a.z, radius, float3{x,y,z});

        ret += val * diff(i,j,k);
      }
    }
  }

  return ret;
}

template <typename Dtype>
float GridMaker::calc_atom_relevance_cpu(const float3& grid_origin, const Grid1f& coord, const Grid<Dtype, 3, false>& density,
    const Grid<Dtype, 3, false>& diff, float radius) const {

  float ret = 0;
  float3 a{coord(0),coord(1),coord(2)}; //atom coordinate

  float r = radius * radius_scale * final_radius_multiple;
  uint2 ranges[3];
  ranges[0] = get_bounds_1d(grid_origin.x, a.x, r);
  ranges[1] = get_bounds_1d(grid_origin.y, a.y, r);
  ranges[2] = get_bounds_1d(grid_origin.z, a.z, r);


  //for every grid point possibly overlapped by this atom
  for (unsigned i = ranges[0].x, iend = ranges[0].y; i < iend;
      ++i) {
    for (unsigned j = ranges[1].x, jend = ranges[1].y; j < jend; ++j) {
      for (unsigned k = ranges[2].x, kend = ranges[2].y; k < kend; ++k) {
        //convert grid point coordinates to angstroms
        float x = grid_origin.x + i * resolution;
        float y = grid_origin.y + j * resolution;
        float z = grid_origin.z + k * resolution;
        float val = 0;
        if(binary)
          val = calc_point<true>(a.x, a.y, a.z, radius, float3{x,y,z});
        else
          val = calc_point<false>(a.x, a.y, a.z, radius, float3{x,y,z});

        if (val > 0) {
          float denseval = density(i,j,k);
          float gridval = diff(i,j,k);
          if(denseval > 0) {
            //weight by contribution to density grid
            ret += gridval*val/denseval;
          } // surely denseval >= val?
        }
      }
    }
  }

  return ret;
}

//cpu backwards
template <typename Dtype>
void GridMaker::backward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    const Grid<Dtype, 4, false>& diff, Grid<Dtype, 2, false>& atom_gradients) const {
  atom_gradients.fill_zero();
  unsigned n = coords.dimension(0);
  if(radii_type_indexed) {
    throw std::invalid_argument("Type indexed radii not supported with index types.");
  }
  if(n != type_index.size()) throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
  if(n != atom_gradients.dimension(0)) throw std::invalid_argument("Gradient dimension doesn't equal number of coordinates");
  if(n != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");
  if(coords.dimension(1) != 3) throw std::invalid_argument("Need x,y,z,r for coord_radius");
  for(unsigned i = 1; i <= 3; i++) {
    if(diff.dimension(i) != dim)
      throw std::invalid_argument("diff does not have correct dimension "+itoa(i)+": "+itoa(diff.dimension(i))+" vs "+itoa(dim));
  }
  float3 grid_origin = get_grid_origin(grid_center);

  for (unsigned i = 0; i < n; ++i) {
    int whichgrid = round(type_index[i]); // this is which atom-type channel of the grid to look at
    if (whichgrid >= 0) {
      if((unsigned)whichgrid >= diff.dimension(0)) {
        throw std::invalid_argument("Incorrect channel size for diff: "+itoa(whichgrid)+" vs "+itoa(diff.dimension(0)));
      }
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

//cpu backwards
template <typename Dtype>
void GridMaker::backward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 2, false>& type_vector, const Grid<float, 1, false>& radii,
    const Grid<Dtype, 4, false>& diff, Grid<Dtype, 2, false>& atom_gradients, Grid<Dtype, 2, false>& type_gradients) const {

  atom_gradients.fill_zero();
  type_gradients.fill_zero();
  unsigned n = coords.dimension(0);
  unsigned ntypes = type_vector.dimension(1);

  if(n != type_vector.dimension(0)) throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
  if(ntypes != diff.dimension(0)) throw std::invalid_argument("Channels in diff doesn't equal number of types");
  if(n != atom_gradients.dimension(0)) throw std::invalid_argument("Atom gradient dimension doesn't equal number of coordinates");
  if(n != type_gradients.dimension(0)) throw std::invalid_argument("Type gradient dimension doesn't equal number of coordinates");
  if(type_gradients.dimension(1) != ntypes) throw std::invalid_argument("Type gradient dimension has wrong number of types");
  if(coords.dimension(1) != 3) throw std::invalid_argument("Need x,y,z,r for coord_radius");

  if(radii_type_indexed) { //radii should be size of types
    if(ntypes != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of types");
  } else { //radii should be size of atoms
    if(n != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");
  }

  float3 grid_origin = get_grid_origin(grid_center);

  for (unsigned i = 0; i < n; ++i) {
    float radius = 0;
    if(!radii_type_indexed)
      radius = radii(i);
    for(unsigned whichgrid = 0; whichgrid < ntypes; whichgrid++) {
      float tmult = type_vector(i,whichgrid);
      if(radii_type_indexed)
        radius = radii(whichgrid);
      if(tmult != 0) {
        float3 agrad = calc_atom_gradient_cpu(grid_origin, coords[i], diff[whichgrid], radius);
        atom_gradients(i,0) += agrad.x*tmult;
        atom_gradients(i,1) += agrad.y*tmult;
        atom_gradients(i,2) += agrad.z*tmult;
      }
      type_gradients(i,whichgrid) = calc_type_gradient_cpu(grid_origin, coords[i], diff[whichgrid], radius);
    }
  }
}

template void GridMaker::backward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 2, false>& type_vector, const Grid<float, 1, false>& radii,
    const Grid<float, 4, false>& diff, Grid<float, 2, false>& atom_gradients, Grid<float, 2, false>& type_gradients) const;
template void GridMaker::backward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 2, false>& type_vector, const Grid<float, 1, false>& radii,
    const Grid<double, 4, false>& diff, Grid<double, 2, false>& atom_gradients, Grid<double, 2, false>& type_gradients) const;


template <typename Dtype>
void GridMaker::backward_gradients(float3 grid_center,  const Grid<float, 2, false>& coords,
    const Grid<float, 2, false>& type_vector, const Grid<float, 1, false>& radii,
    const Grid<Dtype, 4, false>& diff,
    const Grid<Dtype, 2, false>& atom_gradients, const Grid<Dtype, 2, false>& type_gradients,
    Grid<Dtype, 4, false>& diffdiff,
    Grid<Dtype, 2, false>& atom_diffdiff, Grid<Dtype, 2, false>& type_diffdiff) {

  unsigned n = coords.dimension(0);
  unsigned ntypes = type_vector.dimension(1);
  check_vector_args(coords, type_vector, radii, diff);

  if(n != type_vector.dimension(0)) throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
  if(ntypes != diff.dimension(0)) throw std::invalid_argument("Channels in diff doesn't equal number of types");
  if(n != atom_gradients.dimension(0)) throw std::invalid_argument("Atom gradient dimension doesn't equal number of coordinates");
  if(n != type_gradients.dimension(0)) throw std::invalid_argument("Type gradient dimension doesn't equal number of coordinates");
  if(n != atom_diffdiff.dimension(0)) throw std::invalid_argument("Atom gradient gradients dimension doesn't equal number of coordinates");
  if(n != type_diffdiff.dimension(0)) throw std::invalid_argument("Type gradient gradients dimension doesn't equal number of coordinates");

  if(type_gradients.dimension(1) != ntypes) throw std::invalid_argument("Type gradient dimension has wrong number of types");
  if(type_diffdiff.dimension(1) != ntypes) throw std::invalid_argument("Type gradient dimension has wrong number of types");
  if(coords.dimension(1) != 3) throw std::invalid_argument("Need x,y,z,r for coord_radius");

  if(radii_type_indexed) { //radii should be size of types
    if(ntypes != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of types");
  } else { //radii should be size of atoms
    if(n != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");
  }

  if(binary) throw std::invalid_argument("Binary densities not supported");

  atom_diffdiff.fill_zero();
  type_diffdiff.fill_zero(); //note this is the right answer - density is a linear function of amount of type
  diffdiff.fill_zero();

  float3 grid_origin = get_grid_origin(grid_center);

  //there are actually three gradients we need to compute
  //the backward pass is a function of the coordinates, types, and grid gradient G' (everything else is constant)
  // g'(c,t,G') -> (c',t')
  //with respect to the G', this is just the previous multiplier

  //iterate over all atoms
  for (size_t aidx = 0; aidx < n; ++aidx) {
    for (size_t tidx = 0; tidx < ntypes; tidx++) {
      Dtype tmult = type_vector(aidx, tidx); //amount of type for this atom
      if (tmult != 0) {
        Grid<Dtype, 3, false> diffG = diff[tidx];

        float ax = coords(aidx, 0);
        float ay = coords(aidx, 1);
        float az = coords(aidx, 2);

        float3 agrad; //cartesian gradient
        agrad.x = atom_gradients(aidx, 0);
        agrad.y = atom_gradients(aidx, 1);
        agrad.z = atom_gradients(aidx, 2);

        float tgrad = type_gradients(aidx, tidx);

        float radius = 0;
        if(radii_type_indexed) radius = radii(tidx);
        else radius = radii(aidx);

        float ar = radius*radius_scale;
        float densityrad = radius * radius_scale * final_radius_multiple;

        uint2 bounds[3];
        bounds[0] = get_bounds_1d(grid_origin.x, ax, densityrad);
        bounds[1] = get_bounds_1d(grid_origin.y, ay, densityrad);
        bounds[2] = get_bounds_1d(grid_origin.z, az, densityrad);

        float3 adiffdiff{0,0,0};

        //for every grid point possibly overlapped by this atom
        for (size_t i = bounds[0].x, iend = bounds[0].y; i < iend; i++) {
          for (size_t j = bounds[1].x, jend = bounds[1].y; j < jend; j++) {
            for (size_t k = bounds[2].x, kend = bounds[2].y; k < kend; k++) {
              float x = grid_origin.x + i * resolution;
              float y = grid_origin.y + j * resolution;
              float z = grid_origin.z + k * resolution;
              float G = diffG(i,j,k);
              size_t offset = ((((tidx * dim) + i) * dim) + j) * dim + k;

              float dist_x = x - ax;
              float dist_y = y - ay;
              float dist_z = z - az;
              float dist2 = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;
              double dist = sqrt(dist2);

              float agrad_dist = density_grad_dist(dist,ar);
              //in backwards did
              // agrad.x += -(dist_x / dist) * (agrad_dist * gridval)
              // differentiate with respect to gridval
              if(std::isfinite(agrad_dist)) { //overlapping grid position
                float gval = 0.0;
                if(dist > 0) {
                  gval += -(dist_x / dist) * (agrad_dist * agrad.x);
                  gval += -(dist_y / dist) * (agrad_dist * agrad.y);
                  gval += -(dist_z / dist) * (agrad_dist * agrad.z);
                  gval *= tmult;
                }
                //type backwards was just the density value
                float val = calc_point<false>(ax, ay, az, radius, float3{x,y,z});
                gval += val*tgrad;

                *(diffdiff.data() + offset) += gval;

                //now accumulate gradient with respect to atom positions
                adiffdiff.x += atom_density_grad_grad(ax, x, dist, ar)*G*tmult*agrad.x;
                if(G != 0) std::cerr << " " << adiffdiff.x << "\n";
                adiffdiff.x += atom_density_grad_grad_other(ax, x, ay, y, dist, ar)*G*tmult*agrad.y;
                if(G != 0) std::cerr << " " << adiffdiff.x << "\n";
                adiffdiff.x += atom_density_grad_grad_other(ax, x, az, z, dist, ar)*G*tmult*agrad.z;
                if(G != 0) std::cerr << " " << adiffdiff.x << "\n";
                adiffdiff.x += type_grad_grad(ax, x, dist, ar)*G*tgrad;
                if(G != 0) std::cerr << " " << adiffdiff.x << "\n";

                adiffdiff.y += atom_density_grad_grad_other(ay, y, ax, x, dist, ar)*G*tmult*agrad.x;
                adiffdiff.y += atom_density_grad_grad(ay, y, dist, ar)*G*tmult*agrad.y;
                adiffdiff.y += atom_density_grad_grad_other(ay, y, az, z, dist, ar)*G*tmult*agrad.z;
                adiffdiff.y += type_grad_grad(ay, y, dist, ar)*G*tgrad;

                adiffdiff.z += atom_density_grad_grad_other(az, z, ax, x, dist, ar)*G*tmult*agrad.x;
                adiffdiff.z += atom_density_grad_grad_other(az, z, ay, y, dist, ar)*G*tmult*agrad.y;
                adiffdiff.z += atom_density_grad_grad(az, z, dist, ar)*G*tmult*agrad.z;
                adiffdiff.z += type_grad_grad(az, z, dist, ar)*G*tgrad;

                if(G != 0) {
                  std::cerr << "G " << G << "  " << adiffdiff.x <<","<<adiffdiff.y << ","<<adiffdiff.z<<"  " << agrad.x<<","<<agrad.y<<","<<agrad.z << "\n";
                }
              } //if valid grid point
            } //k
          } //j
        } //i
        atom_diffdiff(aidx,0) += adiffdiff.x;
        atom_diffdiff(aidx,1) += adiffdiff.y;
        atom_diffdiff(aidx,2) += adiffdiff.z;

      } //tmult != 0
    } //tidx
  } //aidx

}

template void GridMaker::backward_gradients(float3,  const Grid<float, 2, false>&,
    const Grid<float, 2, false>&, const Grid<float, 1, false>&, const Grid<float, 4, false>&,
    const Grid<float, 2, false>&, const Grid<float, 2, false>&, Grid<float, 4, false>&,
    Grid<float, 2, false>&, Grid<float, 2, false>&);
template void GridMaker::backward_gradients(float3 grid_center,  const Grid<float, 2, false>&,
    const Grid<float, 2, false>&, const Grid<float, 1, false>&, const Grid<double, 4, false>&,
    const Grid<double, 2, false>&, const Grid<double, 2, false>&, Grid<double, 4, false>&,
    Grid<double, 2, false>&, Grid<double, 2, false>&);


template <typename Dtype>
void GridMaker::backward_relevance(float3 grid_center,  const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    const Grid<Dtype, 4, false>& density, const Grid<Dtype, 4, false>& diff,
    Grid<Dtype, 1, false>& relevance) const {

  relevance.fill_zero();
  unsigned n = coords.dimension(0);
  if(n != type_index.size()) throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
  if(coords.dimension(1) != 3) throw std::invalid_argument("Need x,y,z,r for coord_radius");
  if(n != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");

  float3 grid_origin = get_grid_origin(grid_center);

  for (unsigned i = 0; i < n; ++i) {
    int whichgrid = round(type_index[i]); // this is which atom-type channel of the grid to look at
    if (whichgrid >= 0) {
      relevance(i) = calc_atom_relevance_cpu(grid_origin, coords[i], density[whichgrid], diff[whichgrid], radii[i]);
    }
  }
}

template void GridMaker::backward_relevance(float3,  const Grid<float, 2, false>&,
    const Grid<float, 1, false>&, const Grid<float, 1, false>&, const Grid<float, 4, false>&,
    const Grid<float, 4, false>&, Grid<float, 1, false>&) const;
template void GridMaker::backward_relevance(float3,  const Grid<float, 2, false>&,
    const Grid<float, 1, false>&, const Grid<float, 1, false>&, const Grid<double, 4, false>&,
    const Grid<double, 4, false>& , Grid<double, 1, false>& ) const;

}
