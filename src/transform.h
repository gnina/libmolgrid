/*
 * \file transform.h
 *
 *  Stateful transformation of Cartesian coordinates.
 *  Includes capability of generating random transformations.
 */

#ifndef TRANSFORM_H_
#define TRANSFORM_H_
#include <random>
#include <stdexcept>

#include "libmolgrid.h"
#include "quaternion.h"
#include "grid.h"
#include "example.h"

namespace libmolgrid {

/** \brief Stateful transformation of Cartesian coordinates.
 *
 *  Stores a center of rotation, quaternion, and translation.
 *  Can apply transformation forward or backward, with or without
 *  translations.
 */
class Transform {
    Quaternion Q; //rotation
    float3 center; //center of rotation
    float3 translate; //amount to move after rotation

  public:
    Transform() {
      center = make_float3(0, 0, 0);
      translate = make_float3(0, 0, 0);
    }

    Transform(const Quaternion& q,
          float3 c = make_float3(0, 0, 0),
          float3 t = make_float3(0, 0, 0))
        : Q(q), center(c), translate(t) {
    }

    /* \brief Create random transform.
     * @param[in] c  Center of rotation
     * @param[in] random_translate Amount (+/-) to randomly translte
     * @param[in] random_rotate If true, apply random rotation
     */
    Transform(float3 c, float random_translate = 0.0, bool random_rotate = false);

    /* \brief Apply 3D transformation on CPU.   It is safe to transform
     * a grid in-place.
     *
     * @param[in] in Nx3 input grid
     * @param[out] out Nx3 output grid (will be overwritten)
     * @param[in] dotranslate if false only a rotation around the origin is applied.
     * (This is for vector quantities such as gradients and normals).
     */
    template <typename Dtype>
    void forward(const Grid<Dtype, 2, false>& in, Grid<Dtype, 2, false>& out, bool dotranslate=true) const;

    /* \brief Apply 3D transformation on GPU.  Same requirements as CPU version.
     */
    template <typename Dtype>
    __host__ void forward(const Grid<Dtype, 2, true>& in, Grid<Dtype, 2, true>& out, bool dotranslate=true) const;

    /* \brief Apply inverse of 3D transformation on CPU.
     * @param[in] in Nx3 input grid
     * @param[out] out Nx3 output grid (will be overwritten)
     * @param[in] dotranslate if false only the inverse rotation is applied
     */
    template <typename Dtype>
    void backward(const Grid<Dtype, 2, false>& in, Grid<Dtype, 2, false>& out, bool dotranslate=true) const;

    /* \brief Apply 3D transformation on GPU.  Same requirements as CPU version.
     */
    template <typename Dtype>
    __host__ void backward(const Grid<Dtype, 2, true>& in, Grid<Dtype, 2, true>& out, bool dotranslate=true) const;

    const Quaternion& quaternion() const { return Q; }
    float3 rotation_center() const { return center; }
    float3 translation() const { return translate; }
  private:

    // Sanity check grid dimensions and throw exceptions if they are wrong
    template <typename Dtype, bool isCUDA>
    void checkGrids(const Grid<Dtype, 2, isCUDA>& in, const Grid<Dtype, 2, isCUDA>& out) const {
      if(in.dimension(0) != out.dimension(0)) {
        throw std::invalid_argument("Different dimensions and input and output coordinates grids.");
      }
      if(in.dimension(1) != 3) {
        throw std::invalid_argument("Input coordinates are not 3D.");
      }
      if(out.dimension(1) != 3) {
        throw std::invalid_argument("Output coordinates are not 3D.");
      }
    }


};


} /* namespace libmolgrid */

#endif /* TRANSFORM_H_ */
