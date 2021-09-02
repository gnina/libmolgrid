/*
 * grid_interpolater.h
 *
 *  Created on: Nov 16, 2020
 *      Author: dkoes
 */

#ifndef INCLUDE_LIBMOLGRID_GRID_INTERPOLATER_H_
#define INCLUDE_LIBMOLGRID_GRID_INTERPOLATER_H_

#include <vector>
#include <array>
#include <algorithm>
#include <cuda_runtime.h>
#include "libmolgrid/coordinateset.h"
#include "libmolgrid/grid.h"
#include "libmolgrid/example.h"
#include "libmolgrid/transform.h"
#include "libmolgrid/cartesian_grid.h"

namespace libmolgrid {

// Docstring_GridInterpolater
/**
 * \class GridInterpolater
 * Applies a transformation (translation and/or rotation) to a source
 * grid to get an output grid of specified resolutions and dimensions.
 */
class GridInterpolater {
  protected:

    float in_resolution = 0.5;
    float out_resolution = 0.05;
    float in_dimension = 0; //length in Angstroms
    float out_dimension = 0;
    unsigned in_dim = 0; //number of points
    unsigned out_dim = 0;

    // Sanity check grid dimensions and throw exceptions if they are wrong
    template <typename Dtype, bool isCUDA>
    void checkGrids(const Grid<Dtype, 4, isCUDA>& in, const Grid<Dtype, 4, isCUDA>& out) const;

    // setup and initialize texture from in
    cudaTextureObject_t initializeTexture( const Grid<float, 3, true>& in) const;

    //deallocate memory for GPU texture memory
    void clearTexture();

    mutable cudaArray_t cuArray = nullptr; //cache to avoid reallocating

  public:

    /** \brief Construct GridInterpolater
     * @param[in] inres resolution of input grid in Angstroms
     * @param[in] indim dimension of cubic input grid in Angstroms
     * @param[in] outres resolution of output grid in Angstroms
     * @param[in] outdim dimension of cubic output grid in Angstroms
     */
    GridInterpolater(float inres, float indim, float outres, float outdim) :
        in_resolution(inres), out_resolution(outres),
        in_dimension(indim), out_dimension(outdim) {
        in_dim = std::round(in_dimension / in_resolution) + 1;
        out_dim = std::round(out_dimension / out_resolution) + 1;
    }

    virtual ~GridInterpolater() { clearTexture(); }

    ///return resolution of input grid in Angstroms
    CUDA_CALLABLE_MEMBER float get_in_resolution() const { return in_resolution; }
    ///return resolution of output grid in Angstroms
    CUDA_CALLABLE_MEMBER float get_out_resolution() const { return out_resolution; }

    ///set input resolution in Angstroms
    CUDA_CALLABLE_MEMBER void set_in_resolution(float res) {
      in_dim = std::round(in_dimension / in_resolution) + 1;
      in_resolution = res;
      clearTexture();
    }
    ///set output resolution in Angstroms
    CUDA_CALLABLE_MEMBER void set_out_resolution(float res) { out_resolution = res; }

    ///get input dimension in Angstroms
    CUDA_CALLABLE_MEMBER float get_in_dimension() const { return in_dimension; }
    ///get output dimension in Angstroms
    CUDA_CALLABLE_MEMBER float get_out_dimension() const { return out_dimension; }

    ///set input dimension in Angstroms
    CUDA_CALLABLE_MEMBER void set_in_dimension(float d) {
      in_dim = std::round(in_dimension / in_resolution) + 1;
      in_dimension = d;
      clearTexture();
    }
    ///set output dimension in Angstroms
    CUDA_CALLABLE_MEMBER void set_out_dimension(float d) { out_dimension = d;}


    // Docstring_GridInterpolater_forward_1
    /* \brief Interpolate a grid tensor from another grid.
     * Grids and transformation are assumed to have the same center.  Grids must be properly sized.
     * Each grid point of the input grid has the transform applied to its position to determine
     * its location in the output grid.
     * @param[in] in  a CPU 4D grid
     * @param[in] transform  a transformation to apply.  Input grid is rotated about its center and then translated.
     * @param[out] out  a CPU 4D grid
     */
    template <typename Dtype>
    void forward(const Grid<Dtype, 4, false>& in, const Transform& transform, Grid<Dtype, 4, false>& out) const {
      float3 center = transform.get_rotation_center();
      forward(center, in, transform, center, out);
    }

    // Docstring_GridInterpolater_forward_2
    /* \brief Interpolate a grid tensor from another grid.
     * Grids are assumed to have the same center.  Grids must be properly sized.
     * @param[in] in  a GPU 4D grid
     * @param[in] transform  a transformation to apply.  Input grid is rotated about its center and then translated.
     * @param[out] out  a GPU 4D grid
     */
    template <typename Dtype>
    void forward(const Grid<Dtype, 4, true>& in, const Transform& transform, Grid<Dtype, 4, true>& out) const {
      float3 center = transform.get_rotation_center();
      forward(center, in, transform, center, out);
    }

    // Docstring_GridInterpolater_forward_3
    /* \brief Convenience function for randomly interpolating a CPU grid tensor from another grid.
     * Grids are assumed to have the same center.  Grids must be properly sized.
     * @param[in] in  a 4D grid
     * @param[out] out  a 4D grid
     * @param[in] random_translation  maximum amount to randomly translate each coordinate (+/-)
     * @param[in] random_rotation  whether or not to randomly rotate
     */
    template <typename Dtype>
    void forward(const Grid<Dtype, 4, false>& in, Grid<Dtype, 4, false>& out, float random_translation=0, bool random_rotation=false) const {
        Transform t(make_float3(0.0, 0.0, 0.0), random_translation, random_rotation);
        forward(in, t, out);
    }

    // Docstring_GridInterpolater_forward_4
    /* \brief Convenience function for randomly interpolating a CUDA grid tensor from another grid.
     * Grids are assumed to have the same center.  Grids must be properly sized.
     * @param[in] in  a 4D grid
     * @param[out] out  a 4D grid
     * @param[in] random_translation  maximum amount to randomly translate each coordinate (+/-)
     * @param[in] random_rotation  whether or not to randomly rotate
     */
    template <typename Dtype>
    void forward(const Grid<Dtype, 4, true>& in, Grid<Dtype, 4, true>& out, float random_translation=0, bool random_rotation=false) const {
        Transform t(make_float3(0.0, 0.0, 0.0), random_translation, random_rotation);
        forward(in, t, out);
    }

    // Docstring_GridInterpolater_forward_5
    /* \brief Interpolate a grid tensor from another grid.
     * Grids and transformation are assumed to have the same center.  Grids must be properly sized.
     * Each grid point of the input grid has the transform applied to its position to determine
     * its location in the output grid.
     * @param[in] in_center  center of input grid
     * @param[in] in  a CPU 4D grid
     * @param[in] transform  a transformation to apply.  Input grid is rotated about its center and then translated.
     * @param[in] out_center center of output grid
     * @param[out] out  a CPU 4D grid
     */
    template <typename Dtype>
    void forward(float3 in_center, const Grid<Dtype, 4, false>& in, const Transform& transform, float3 out_center, Grid<Dtype, 4, false>& out) const;

    // Docstring_GridInterpolater_forward_6
    /* \brief Interpolate a grid tensor from another grid.
     * Grids are assumed to have the same center.  Grids must be properly sized.
     * @param[in] in_center  center of input grid
     * @param[in] in  a GPU 4D grid
     * @param[in] transform  a transformation to apply.  Input grid is rotated about its center and then translated.
     * @param[in] out_center center of output grid
     * @param[out] out  a GPU 4D grid
     */
    template <typename Dtype>
    void forward(float3 in_center, const Grid<Dtype, 4, true>& in, const Transform& transform, float3 out_center, Grid<Dtype, 4, true>& out) const;

    //TODO: backwards

    template <typename Dtype, bool isCUDA>
    CUDA_CALLABLE_MEMBER Dtype get_pt(const Grid<Dtype, 3, isCUDA>& in, int x, int y, int z) const;

    template <typename Dtype, bool isCUDA>
    CUDA_CALLABLE_MEMBER Dtype interpolate(const Grid<Dtype, 3, isCUDA>& in, float3 gridpt) const;
};

// return grid coordinates (not rounded) for Cartesian coordinates
inline CUDA_CALLABLE_MEMBER float3 cart2grid(float3 origin, float resolution, float x, float y, float z) {
    float3 pt = { (x-origin.x)/resolution, (y-origin.y)/resolution, (z-origin.z)/resolution };
    return pt;
}

// return Cartesian coordinates of provided grid position
inline CUDA_CALLABLE_MEMBER float3 grid2cart(float3 origin, float resolution, unsigned i, unsigned j, unsigned k) {
    float3 pt = {origin.x+i*resolution,origin.y+j*resolution,origin.z+k*resolution};
    return pt;
}


} /* namespace libmolgrid */



#endif /* INCLUDE_LIBMOLGRID_GRID_INTERPOLATER_H_ */
