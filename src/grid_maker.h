/*
 * grid_maker.h
 *
 *  Grid generation form atomic data.
 *  Created on: Mar 26, 2019
 *      Author: dkoes
 */

#ifndef GRID_MAKER_H_
#define GRID_MAKER_H_

#include <vector>
#include <array>
#include <algorithm>
#include <cuda_runtime.h>
#include "coordinateset.h"
#include "grid.h"

namespace libmolgrid {

/**
 * \class GridMaker
 * Populates a grid with atom density values that correspond to atoms in a
 * CoordinateSet and accumulates atomic gradients from the grid gradients. 
 * It stores state about universal grid settings. In functions that map from
 * atomic coordinates to grids and vice versa (e.g. forward and backward), it
 * must be passed the grid_center (which may have changed due to
 * transformations performed directly on the atom coordinates externally to
 * this class)
 */
class GridMaker {
  protected:
    float radiusmultiple; // at what multiple of the atomic radius does the atom density go to 0
    float resolution; // grid spacing
    float dimension; // grid side length in Angstroms
    size_t dim; // grid width in points
    bool binary; // use binary occupancy instead of real-valued atom density

  public:

    GridMaker();
    virtual ~GridMaker() {}

    float3 getGridDims() const { 
      float3 dims = make_float3(dim, dim, dim); 
      return dims;
    }

    /* \brief Use externally specified grid_center to determine where grid begins and
     * end. Used before translating between cartesian coords and grids.
     * @param[in] grid center
     * @param[out] grid bounds
     */
    float3 getGridOrigin(float3 grid_center) const {
      float half = dimension/2.0;
      float3 grid_origin;
      grid_origin.x = grid_center.x - half;
      grid_origin.y = grid_center.y - half;
      grid_origin.z = grid_center.z - half;
      return grid_origin;
    }

    /* \brief Find grid indices in one dimension that bound an atom's density. 
     * @param[in] grid min coordinate in a given dimension
     * @param[in] atom coordinate in the same dimension
     * @param[in] atomic density radius (N.B. this is not the atomic radius)
     * @param[out] indices of grid points in the same dimension that could
     * possibly overlap atom density
     */
    std::pair<size_t, size_t> getBounds_1D(const float grid_origin, float coord, float densityrad)  const {
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

    /* \brief Calculate atom density at a grid point.
     * @param[in] atomic coords
     * @param[in] atomic radius
     * @param[in] grid point coords
     * @param[out] atom density
     */
    CUDA_CALLABLE_MEMBER float calcPoint(const float3& coords, double ar, const float3& grid_coords) const {
      float dx = grid_coords.x - coords.x;
      float dy = grid_coords.y - coords.y;
      float dz = grid_coords.z - coords.z;

      float rsq = dx * dx + dy * dy + dz * dz;
      if (binary) {
        //is point within radius?
        if (rsq < ar * ar)
          return 1.0;
        else
          return 0.0;
      } else {
        //For non-binary density we want a Gaussian where 2 std occurs at the
        //radius, after which it becomes quadratic.  
        //The quadratic is fit to have both the same value and first derivative
        //at the cross over point and a value and derivative of zero at
        //1.5*radius 
        //FIXME wrong for radiusmultiple != 1.5
        double dist = sqrt(rsq);
        if (dist >= ar * radiusmultiple) {
          return 0.0;
        } else
          if (dist <= ar) {
            //return gaussian
            float h = 0.5 * ar;
            float ex = -dist * dist / (2 * h * h);
            return exp(ex);
          } else //return quadratic
          {
            float h = 0.5 * ar;
            float eval = 1.0 / (M_E * M_E); //e^(-2)
            float q = dist * dist * eval / (h * h) - 6.0 * eval * dist / h
                + 9.0 * eval;
            return q;
          }
      }
    }

    /* \brief Generate grid tensor from CPU atomic data.  Grid must be properly sized.
     * @param[in] center of grid
     * @param[in] coordinate set
     * @param[out] a 4D grid
     */
    template <typename Dtype>
    void forward(float3 grid_center, const CoordinateSet& in, Grid<Dtype, 4, false>& out) const {
      if(in.has_indexed_types()) {
        forward(grid_center, in.coord.cpu(), in.type_index.cpu(), in.radius.cpu(), out);
      } else {
        throw std::invalid_argument("Type vector gridding not implemented yet");
      }
    }

    /* \brief Generate grid tensor from GPU atomic data.  Grid must be properly sized.
     * @param[in] center of grid
     * @param[in] coordinate set
     * @param[out] a 4D grid
     */
    template <typename Dtype>
    void forward(float3 grid_center, const CoordinateSet& in, Grid<Dtype, 4, true>& out) const {
      if(in.has_indexed_types()) {
        forward(grid_center, in.coord.gpu(), in.type_index.gpu(), in.radius.gpu(), out);
      } else {
        throw std::invalid_argument("Type vector gridding not implemented yet");
      }
    }

    /* \brief Generate grid tensor from CPU atomic data.  Grid must be properly sized.
     * @param[in] center of grid
     * @param[in] coordinates (Nx3)
     * @param[in] type indices (N integers stored as floats)
     * @param[in] radii (N)
     * @param[out] a 4D grid
     */
    template <typename Dtype>
    void forward(float3 grid_center, const Grid<float, 2, false>& coords,
        const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
        Grid<Dtype, 4, false>& out) const {
      float3 grid_origin = getGridOrigin(grid_center);
      size_t natoms = coords.dimension(0);
      for (size_t aidx=0; aidx<natoms; ++aidx) {
        float x = coords(aidx, 0);
        float y = coords(aidx, 1);
        float z = coords(aidx, 2);
        float atype = type_index(aidx);
        float radius = radii(aidx);
        float densityrad = radius * radiusmultiple;

        std::array<std::pair<size_t,size_t>,3> bounds;
        bounds[0] = getBounds_1D(grid_origin.x, coords(aidx,0), densityrad);
        bounds[1] = getBounds_1D(grid_origin.y, coords(aidx,1), densityrad);
        bounds[2] = getBounds_1D(grid_origin.z, coords(aidx,2), densityrad);

        //for every grid point possibly overlapped by this atom
        for (size_t i = bounds[0].first, iend = bounds[0].second; i < iend; i++) {
          for (size_t j = bounds[1].first, jend = bounds[1].second; j < jend;
              j++) {
            for (size_t k = bounds[2].first, kend = bounds[2].second;
                k < kend; k++) {
              float3 grid_coords;
              float3 acoords = make_float3(x, y, z);
              grid_coords.x = grid_origin.x + i * resolution;
              grid_coords.y = grid_origin.y + j * resolution;
              grid_coords.z = grid_origin.z + k * resolution;
              float val = calcPoint(acoords, radius, grid_coords);

              if (binary) {
                if (val != 0) 
                  out(atype, i, j, k) = 1.0;
              }
              else {
                  out(atype, i, j, k) += val;
              }

            }
          }
        }
      }
    }

    /* \brief Generate grid tensor from GPU atomic data.  Grid must be properly sized.
     * @param[in] center of grid
     * @param[in] coordinates (Nx3)
     * @param[in] type indices (N integers stored as floats)
     * @param[in] radii (N)
     * @param[out] a 4D grid
     */
    template <typename Dtype>
    void forward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
        Grid<Dtype, 4, true>& out) const;

    /* \brief The GPU forward code path launches a kernel (forward_gpu) that
     * sets the grid values in two steps: first each thread cooperates with the
     * other threads in its block to determine which atoms could possibly
     * overlap them. Then, working from this significantly reduced atom set,
     * they actually check whether they are overlapped by an atom and update
     * their density accordingly. atomOverlapsBlock is a helper for generating
     * the reduced array of possibly relevant atoms.
     * @param[in] atom index to check
     * @param[in] grid origin
     * @param[in] coordinates (Nx3)
     * @param[in] type indices (N integers stored as floats)
     * @param[in] radii (N)
     * @param[out] 1 if atom could overlap block, 0 if not
     */
    CUDA_DEVICE_MEMBER
    unsigned atomOverlapsBlock(unsigned aidx, float3 grid_origin, 
        const Grid<float, 2, true>& coords, const Grid<float, 1, true>& type_index, 
        const Grid<float, 1, true>& radii);

    /* \brief The function that actually updates the voxel density values. 
     * @param[in] number of possibly relevant atoms
     * @param[in] grid origin
     * @param[in] coordinates (Nx3)
     * @param[in] type indices (N integers stored as floats)
     * @param[in] radii (N)
     * @param[out] a 4D grid
     */
    template <typename Dtype>
    CUDA_DEVICE_MEMBER void set_atoms(unsigned natoms, float3& grid_origin, 
        const Grid<float, 2, true>& coords, const Grid<float, 1, true>& type_index, 
        const Grid<float, 1, true>& radii, Grid<Dtype, 4, true>& out);
};

} /* namespace libmolgrid */

#endif /* GRID_MAKER_H_ */
