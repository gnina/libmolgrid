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

#include "libmolgrid/libmolgrid.h"
#include "libmolgrid/quaternion.h"
#include "libmolgrid/grid.h"
#include "libmolgrid/example.h"

namespace libmolgrid {

// Docstring_Transform
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

    // Docstring_Transform_constructor
    /* \brief Create random transform.
     * @param[in] c  Center of rotation
     * @param[in] random_translate Amount (+/-) to randomly translte
     * @param[in] random_rotate If true, apply random rotation
     */
    Transform(float3 c, float random_translate = 0.0, bool random_rotate = false);

    // Docstring_Transform_forward_1
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

    // Docstring_Transform_forward_2
    /* \brief Apply 3D transformation to Example.   It is safe to transform in-place
     *
     * @param[in] input example
     * @param[out] output example with same dimensions
     * @param[in] dotranslate if false only a rotation around the origin is applied.
     * (This is for vector quantities such as gradients and normals).
     */
    void forward(const Example& in, Example& out, bool dotranslate=true) const;

    // Docstring_Transform_forward_3
    /* \brief Apply 3D transformation to CoordinateSet.   It is safe to transform in-place
     *
     * @param[in] input coords
     * @param[out] output coords with same dimensions
     * @param[in] dotranslate if false only a rotation around the origin is applied.
     * (This is for vector quantities such as gradients and normals).
     */
    void forward(const CoordinateSet& in, CoordinateSet& out, bool dotranslate=true) const;

    // Docstring_Transform_forward_4
    /* \brief Apply 3D transformation on GPU.  It is safe to transform a grid
     * in-place.
     *
     * @param[in] in Nx3 input grid
     * @param[out] out Nx3 output grid (will be overwritten)
     * @param[in] dotranslate if false only a rotation around the origin is applied.
     * (This is for vector quantities such as gradients and normals).
     */
    template <typename Dtype>
    __host__ void forward(const Grid<Dtype, 2, true>& in, Grid<Dtype, 2, true>& out, bool dotranslate=true) const;

    // Docstring_Transform_backward_1
    /* \brief Apply inverse of 3D transformation on CPU.
     * @param[in] in Nx3 input grid
     * @param[out] out Nx3 output grid (will be overwritten)
     * @param[in] dotranslate if false only the inverse rotation is applied
     */
    template <typename Dtype>
    void backward(const Grid<Dtype, 2, false>& in, Grid<Dtype, 2, false>& out, bool dotranslate=true) const;

    // Docstring_Transform_backward_2
    /* \brief Apply inverse of 3D transformation on GPU.
     * @param[in] in Nx3 input grid
     * @param[out] out Nx3 output grid (will be overwritten)
     * @param[in] dotranslate if false only the inverse rotation is applied
     */
    template <typename Dtype>
    __host__ void backward(const Grid<Dtype, 2, true>& in, Grid<Dtype, 2, true>& out, bool dotranslate=true) const;

    const Quaternion& get_quaternion() const { return Q; }
    float3 get_rotation_center() const { return center; }
    float3 get_translation() const { return translate; }

    void set_quaternion(const Quaternion& q) { Q = q; }
    void set_rotation_center(float3 c) { center = c; }
    void set_translation(float3 t) { translate = t; }

    /// transformation does not change inputs
    bool is_identity() const {
      return Q == Quaternion() && translate.x == 0 && translate.y == 0 && translate.z == 0;
    }

  private:

    // Sanity check grid dimensions and throw exceptions if they are wrong
    template <typename Dtype, bool isCUDA>
    void checkGrids(const Grid<Dtype, 2, isCUDA>& in, const Grid<Dtype, 2, isCUDA>& out) const {
      if(in.dimension(0) != out.dimension(0)) {
        throw std::invalid_argument("Different dimensions and input and output coordinates grids.");
      }
      if(in.dimension(1) != 3) {
        throw std::invalid_argument("Input coordinates wrong dimension.");
      }
      if(out.dimension(1) != 3) {
        throw std::invalid_argument("Output coordinates are wrong dimension.");
      }
      if(in.data() == nullptr) {
        throw std::invalid_argument("Input coordinates missing memory");
      }
      if(out.data() == nullptr) {
        throw std::invalid_argument("Output coordinates missing memory");
      }
    }


};


} /* namespace libmolgrid */

#endif /* TRANSFORM_H_ */
