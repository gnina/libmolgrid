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
#include "libmolgrid/coordinateset.h"
#include "libmolgrid/grid.h"
#include "libmolgrid/example.h"
#include "libmolgrid/transform.h"

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
    float resolution; // grid spacing
    float dimension; // grid side length in Angstroms
    float radiusmultiple; // at what multiple of the atomic radius does the atom density go to 0
    bool binary; // use binary occupancy instead of real-valued atom density
    size_t dim; // grid width in points

    template<typename Dtype, bool isCUDA>
    void check_index_args(const Grid<float, 2, isCUDA>& coords,
        const Grid<float, 1, isCUDA>& type_index, const Grid<float, 1, isCUDA>& radii,
        Grid<Dtype, 4, isCUDA>& out) const;
  public:

    GridMaker(float res = 0, float d = 0, float rm = 1.5, bool bin = false) :
      resolution(res), dimension(d), radiusmultiple(rm), binary(bin) {
        initialize(res, d, rm, bin);
      }

    virtual ~GridMaker() {}

    void initialize(float res, float d, float rm, bool bin = false) {
      resolution = res;
      dimension = d;
      radiusmultiple = rm;
      dim = ::round(dimension / resolution) + 1;
    }

    float3 get_grid_dims() const {
      return make_float3(dim, dim, dim);
    }

    float get_resolution() const { return resolution; }
    void set_resolution(float res) { resolution = res; dim = ::round(dimension / resolution) + 1; }

    float get_dimension() const { return dimension; }
    void set_dimension(float d) { dimension = d; dim = ::round(dimension / resolution) + 1; }

    bool get_binary() const { return binary; }
    void set_binary(bool b) { binary = b; }

    /* \brief Use externally specified grid_center to determine where grid begins.
     * Used for translating between cartesian coords and grids.
     * @param[in] grid center
     * @param[out] grid bounds
     */
    float3 get_grid_origin(const float3& grid_center) const;

    template <typename Dtype>
    CUDA_DEVICE_MEMBER void zero_grid(Grid<Dtype, 4, true>& grid);

    /* \brief Find grid indices in one dimension that bound an atom's density.
     * @param[in] grid min coordinate in a given dimension
     * @param[in] atom coordinate in the same dimension
     * @param[in] atomic density radius (N.B. this is not the atomic radius)
     * @param[out] indices of grid points in the same dimension that could
     * possibly overlap atom density
     */
    std::pair<size_t, size_t> get_bounds_1d(const float grid_origin, float coord,
        float densityrad)  const;

    /* \brief Calculate atom density at a grid point.
     * @param[in] atomic coords
     * @param[in] atomic radius
     * @param[in] grid point coords
     * @param[out] atom density
     */
    CUDA_CALLABLE_MEMBER float calc_point(const float3& coords, double ar,
        const float3& grid_coords) const;


    /* \brief Generate grid tensor from atomic data.  Grid (CPU) must be properly sized.
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

    /* \brief Generate grid tensor from atomic data.  Grid (GPU) must be properly sized.
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

    /* \brief Generate grid tensor from an example while applying a transformation.
     * The center specified in the transform will be used as the grid center.
     *
     * @param[in] ex example
     * @param[in] transform transformation to apply
     * @param[out] out a 4D grid
     */
    template <typename Dtype, bool isCUDA>
    void forward(const Example& in, const Transform& transform, Grid<Dtype, 4, isCUDA>& out) const;

    /* \brief Generate grid tensor from an example.
     * Coordinates may be optionally translated/rotated.  Do not use this function
     * if it is desirable to retain the transformation used (e.g., when backpropagating).
     *
     * @param[in] ex example
     * @param[in] transform transformation to apply
     * @param[out] out a 4D grid
     * @param[in] random_translation  maximum amount to randomly translate each coordinate (+/-)
     * @param[in] random_rotation whether or not to randomly rotate
     * @param[in] center grid center to use, if not provided will use center of the last coordinate set before transformation
     */
    template <typename Dtype, bool isCUDA>
    void forward(const Example& in, Grid<Dtype, 4, isCUDA>& out,
        float random_translation=0.0, bool random_rotation = false,
        const float3& center = make_float3(INFINITY, INFINITY, INFINITY)) const;

    /* \brief Generate grid tensor from a vector of examples, as provided by ExampleProvider.next_batch.
     * Coordinates may be optionally translated/rotated.  Do not use this function
     * if it is desirable to retain the transformation used (e.g., when backpropagating).
     * The center of the last coordinate set before transformation
     * will be used as the grid center.
     *
     * @param[in] ex example
     * @param[in] transform transformation to apply
     * @param[out] out a 4D grid
     * @param[in] random_translation  maximum amount to randomly translate each coordinate (+/-)
     * @param[in] random_rotation whether or not to randomly rotate
     */
    template <typename Dtype, bool isCUDA>
    void forward(const std::vector<Example>& in, Grid<Dtype, 5, isCUDA>& out, float random_translation=0.0, bool random_rotation = false) const {
      if(in.size() != out.dimension(0)) throw std::out_of_range("output grid dimension does not match size of example vector");
      for(unsigned i = 0, n = in.size(); i < n; i++) {
        Grid<Dtype, 4, isCUDA> g(out[i]);
        forward<Dtype,isCUDA>(in[i],g, random_translation, random_rotation);
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
        Grid<Dtype, 4, false>& out) const;

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

    /* \brief Generate atom and type gradients from grid gradients. (CPU)
     * Must provide atom coordinates that defined the original grid in forward
     * Vector types are required.
     * @param[in] center of grid
     * @param[in] in coordinate set
     * @param[in] diff a 4D grid of gradients
     * @param[out] atomic_gradients vector quantities for each atom
     * @param[out] type_gradients only set if input has type vectors
     */
    template <typename Dtype>
    void backward(float3 grid_center, const CoordinateSet& in, const Grid<Dtype, 4, false>& diff,
        Grid<Dtype, 2, false>& atomic_gradients, Grid<Dtype, 2, false>& type_gradients) const {
      if(in.has_vector_types()) {
        backward(grid_center, in.coord.cpu(), in.type_vector.cpu(), in.radius.cpu(), diff, atomic_gradients, type_gradients);
      } else {
        throw std::invalid_argument("Vector types missing from coordinate set");
      }
    }

    /* \brief Generate atom gradients from grid gradients. (CPU)
     * Must provide atom coordinates that defined the original grid in forward
     * Index types are required
     * @param[in] center of grid
     * @param[in] in coordinate set
     * @param[in] diff a 4D grid of gradients
     * @param[out] atomic_gradients vector quantities for each atom
     */
    template <typename Dtype>
    void backward(float3 grid_center, const CoordinateSet& in, const Grid<Dtype, 4, false>& diff,
        Grid<Dtype, 2, false>& atomic_gradients) const {
      if(in.has_indexed_types()) {
        backward(grid_center, in.coord.cpu(), in.type_index.cpu(), in.radius.cpu(), diff, atomic_gradients);
      } else {
        throw std::invalid_argument("Index types missing from coordinate set"); //could setup dummy types here
      }
    }

    /* \brief Generate atom and type gradients from grid gradients. (GPU)
     * Must provide atom coordinates that defined the original grid in forward
     * Vector types are required.
     * @param[in] center of grid
     * @param[in] in coordinate set
     * @param[in] diff a 4D grid of gradients
     * @param[out] atomic_gradients vector quantities for each atom
     * @param[out] type_gradients only set if input has type vectors
     */
    template <typename Dtype>
    void backward(float3 grid_center, const CoordinateSet& in, const Grid<Dtype, 4, true>& diff,
        Grid<Dtype, 2, true>& atomic_gradients, Grid<Dtype, 2, true>& type_gradients) const {
      if(in.has_vector_types()) {
        backward(grid_center, in.coord.gpu(), in.type_vector.gpu(), in.radius.gpu(), diff, atomic_gradients, type_gradients);
      } else {
        throw std::invalid_argument("Vector types missing from coordinate set");
      }
    }

    /* \brief Generate atom gradients from grid gradients. (GPU)
     * Must provide atom coordinates that defined the original grid in forward
     * Index types are required.
     * @param[in] center of grid
     * @param[in] in coordinate set
     * @param[in] diff a 4D grid of gradients
     * @param[out] atomic_gradients vector quantities for each atom
     */
    template <typename Dtype>
    void backward(float3 grid_center, const CoordinateSet& in, const Grid<Dtype, 4, true>& diff,
        Grid<Dtype, 2, true>& atomic_gradients) const {
      if(in.has_indexed_types()) {
        backward(grid_center, in.coord.gpu(), in.type_index.gpu(), in.radius.gpu(), diff, atomic_gradients);
      } else {
        throw std::invalid_argument("Index types missing from coordinate set");
      }
    }

    /* \brief Generate atom gradients from grid gradients. (CPU)
     * Must provide atom coordinates, types, and radii that defined the original grid in forward
     * @param[in] center of grid
     * @param[in] coordinates (Nx3)
     * @param[in] type indices (N integers stored as floats)
     * @param[in] radii (N)
     * @param[in] diff a 4D grid of gradients
     * @param[out] atomic_gradients vector quantities for each atom
     */
    template <typename Dtype>
    void backward(float3 grid_center, const Grid<float, 2, false>& coords,
        const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
        const Grid<Dtype, 4, false>& diff, Grid<Dtype, 2, false>& atom_gradients);

    /* \brief Generate atom gradients from grid gradients. (GPU)
     * Must provide atom coordinates, types, and radii that defined the original grid in forward
     * @param[in] center of grid
     * @param[in] coordinates (Nx3)
     * @param[in] type indices (N integers stored as floats)
     * @param[in] radii (N)
     * @param[in] diff a 4D grid of gradients
     * @param[out] atomic_gradients vector quantities for each atom
     */
    template <typename Dtype>
    void backward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
        const Grid<Dtype, 4, true>& grid, Grid<Dtype, 2, true>& atom_gradients);

    /* \brief Generate atom and type gradients from grid gradients. (CPU)
     * Must provide atom coordinates, types, and radii that defined the original grid in forward
     * @param[in] center of grid
     * @param[in] coordinates (Nx3)
     * @param[in] type vectors (NxT)
     * @param[in] radii (N)
     * @param[in] diff a 4D grid of gradients
     * @param[out] atomic_gradients vector quantities for each atom
     * @param[out] type_gradients vector quantities for each atom
     */
    template <typename Dtype>
    void backward(float3 grid_center, const Grid<float, 2, false>& coords,
        const Grid<float, 2, false>& type_vectors, const Grid<float, 1, false>& radii,
        const Grid<Dtype, 4, false>& diff,
        Grid<Dtype, 2, false>& atom_gradients, Grid<Dtype, 2, false>& type_gradients) {
      throw std::runtime_error("Vector type gradient calculation not implemented yet");
    }

    /* \brief Generate atom gradients from grid gradients. (GPU)
     * Must provide atom coordinates, types, and radii that defined the original grid in forward
     * @param[in] center of grid
     * @param[in] coordinates (Nx3)
     * @param[in] type vectors (NxT)
     * @param[in] radii (N)
     * @param[in] diff a 4D grid of gradients
     * @param[out] atomic_gradients vector quantities for each atom
     * @param[out] type_gradients vector quantities for each atom
     *
     */
    template <typename Dtype>
    void backward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 2, true>& type_vectors, const Grid<float, 1, true>& radii,
        const Grid<Dtype, 4, true>& grid,
        Grid<Dtype, 2, true>& atom_gradients,  Grid<Dtype, 2, true>& type_gradients) {
      throw std::runtime_error("Vector type gradient calculation not implemented yet");
    }

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
    unsigned atom_overlaps_block(unsigned aidx, float3& grid_origin,
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
    CUDA_DEVICE_MEMBER void set_atoms(size_t natoms, float3& grid_origin,
        const Grid<float, 2, true>& coords, const Grid<float, 1, true>& type_index,
        const Grid<float, 1, true>& radii, Grid<Dtype, 4, true>& out);
};

//specify what instantiations are in grid_maker.cpp
extern template void GridMaker::forward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    Grid<float, 4, false>& out) const;
extern template void GridMaker::forward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    Grid<double, 4, false>& out) const;

extern template void GridMaker::forward(float3 grid_center, const Grid<float, 2, true>& coords,
    const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
    Grid<float, 4, true>& out) const;
extern template void GridMaker::forward(float3 grid_center, const Grid<float, 2, true>& coords,
    const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
    Grid<double, 4, true>& out) const;

extern template void GridMaker::forward(const Example& in, Grid<float, 4, false>& out,
    float random_translation, bool random_rotation, const float3& center) const;
extern template void GridMaker::forward(const Example& in, Grid<float, 4, true>& out,
    float random_translation, bool random_rotation, const float3& center) const;
extern template void GridMaker::forward(const Example& in, Grid<double, 4, false>& out,
    float random_translation, bool random_rotation, const float3& center) const;
extern template void GridMaker::forward(const Example& in, Grid<double, 4, true>& out,
    float random_translation, bool random_rotation, const float3& center) const;

extern template void GridMaker::forward(const Example& in, const Transform& transform, Grid<float, 4, false>& out) const;
extern template void GridMaker::forward(const Example& in, const Transform& transform, Grid<float, 4, true>& out) const;
extern template void GridMaker::forward(const Example& in, const Transform& transform, Grid<double, 4, false>& out) const;
extern template void GridMaker::forward(const Example& in, const Transform& transform, Grid<double, 4, true>& out) const;

extern template void GridMaker::forward(const std::vector<Example>& in, Grid<float, 5, false>& out,
  float random_translation, bool random_rotation) const;
extern template void GridMaker::forward(const std::vector<Example>& in, Grid<float, 5, true>& out,
    float random_translation, bool random_rotation) const;
extern template void GridMaker::forward(const std::vector<Example>& in, Grid<double, 5, false>& out,
  float random_translation, bool random_rotation) const;
extern template void GridMaker::forward(const std::vector<Example>& in, Grid<double, 5, true>& out,
    float random_translation, bool random_rotation) const;


extern template void GridMaker::backward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    const Grid<float, 4, false>& diff, Grid<float, 2, false>& atom_gradients);
extern template void GridMaker::backward(float3 grid_center, const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    const Grid<double, 4, false>& diff, Grid<double, 2, false>& atom_gradients);
extern template void GridMaker::backward(float3 grid_center, const Grid<float, 2, true>& coords,
    const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
    const Grid<float, 4, true>& grid, Grid<float, 2, true>& atom_gradients);
extern template void GridMaker::backward(float3 grid_center, const Grid<float, 2, true>& coords,
    const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
    const Grid<double, 4, true>& grid, Grid<double, 2, true>& atom_gradients);


extern template void GridMaker::check_index_args(const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    Grid<float, 4, false>& out) const;
extern template void GridMaker::check_index_args(const Grid<float, 2, true>& coords,
    const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
    Grid<float, 4, true>& out) const;
extern template void GridMaker::check_index_args(const Grid<float, 2, false>& coords,
    const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
    Grid<double, 4, false>& out) const;
extern template void GridMaker::check_index_args(const Grid<float, 2, true>& coords,
    const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
    Grid<double, 4, true>& out) const;
} /* namespace libmolgrid */

#endif /* GRID_MAKER_H_ */
