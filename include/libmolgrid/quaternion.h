/**
 * quaternion.h
 *
 */

#ifndef QUATERNION_H_
#define QUATERNION_H_

#include <cuda_runtime.h>

namespace libmolgrid {

// Docstring_Quaternion
/** \brief A CUDA friendly quaternion class.  Single precision only
 * for maximum CUDA performance.
 */
class Quaternion {
  public:
    typedef float fl;

  protected:
    fl a;
    fl b;
    fl c;
    fl d;

  public:
    __host__ __device__ Quaternion()
        : a(1), b(0), c(0), d(0) {
    }

    /// Construct quaternion with real then unreal components
    __host__ __device__ Quaternion(fl A, fl B, fl C, fl D)
        : a(A), b(B), c(C), d(D) {
    }

    /// Return the real or scalar part.
    __host__ __device__ inline fl R_component_1() const {
      return a;
    }

    /// Return unreal[0]
    __host__ __device__ inline fl R_component_2() const {
      return b;
    }

    /// Return unreal[1]
    __host__ __device__ inline fl R_component_3() const {
      return c;
    }

    /// Return unreal[2]
    __host__ __device__ inline fl R_component_4() const {
      return d;
    }

    /// In-place multiplication by scalar (element-wise)
    __host__ __device__ Quaternion& operator *=(const fl &r) {
      a *= r;
      b *= r;
      c *= r;
      d *= r;
      return *this;
    }

    /// In-place division by scalar (element-wise)
    __host__ __device__ Quaternion& operator /=(const fl &r) {
      a /= r;
      b /= r;
      c /= r;
      d /= r;
      return *this;
    }

    /// Division by scalar (element-wise)
    __host__  __device__ Quaternion operator/(const fl &r) {
      return Quaternion(a/r,b/r,c/r,d/r);
    }

    /// Quaternion multiplication
    __host__  __device__
     inline Quaternion operator*(const Quaternion& r) const {
      const fl ar = r.R_component_1();
      const fl br = r.R_component_2();
      const fl cr = r.R_component_3();
      const fl dr = r.R_component_4();

      return Quaternion(+a * ar - b * br - c * cr - d * dr,
          +a * br + b * ar + c * dr - d * cr, +a * cr - b * dr + c * ar + d * br,
          +a * dr + b * cr - c * br + d * ar);
    }

    /// Quaternion in-place multiplication
    __host__ __device__ Quaternion& operator*=(const Quaternion& r) {
      *this = *this * r;
      return *this;
    }

    ///check bit level equality
    __host__ __device__ bool operator==(const Quaternion& r) const {
      return a == r.a && b == r.b && c == r.c && d == r.d;
    }

    /// Quaternion divison
    __host__ __device__
    inline Quaternion operator/(const Quaternion& r) {
      const fl ar = r.R_component_1();
      const fl br = r.R_component_2();
      const fl cr = r.R_component_3();
      const fl dr = r.R_component_4();

      fl denominator = ar * ar + br * br + cr * cr + dr * dr;

      fl at = (+a * ar + b * br + c * cr + d * dr) / denominator;
      fl bt = (-a * br + b * ar - c * dr + d * cr) / denominator;
      fl ct = (-a * cr + b * dr + c * ar - d * br) / denominator;
      fl dt = (-a * dr - b * cr + c * br + d * ar) / denominator;

      return Quaternion(at,bt,ct,dt);
    }

    /// Quaternion in-place divison
    __host__ __device__ Quaternion operator/=(const Quaternion &r) {
      *this = *this / r;
      return *this;
    }

    /// Conjugate
    __host__ __device__ inline Quaternion conj() const {
      return Quaternion(+a, -b, -c,-d);
    }

    __host__ __device__
    float real() const {
      return R_component_1();
    }

    /// The Cayley norm - the square of the Euclidean norm
    __host__ __device__
    inline float norm() const {
      return a*a+b*b+c*c+d*d;
    }

    /// Rotation point (x,y,z) using this quaternion.
    __host__    __device__ float3 rotate(fl x, fl y, fl z) const {
      Quaternion p(0, x, y, z);
      p = *this * p * (conj() / norm());
      return make_float3(p.R_component_2(), p.R_component_3(), p.R_component_4());
    }

    /// Rotate around the provided center and translate
    __host__ __device__ inline float3 transform(fl x, fl y, fl z, float3 center, float3 translate) const {
      float3 pt = rotate(x - center.x, y - center.y, z - center.z);
      x = pt.x + center.x + translate.x;
      y = pt.y + center.y + translate.y;
      z = pt.z + center.z + translate.z;

      return make_float3(x,y,z);
    }


    /// Return inverse
    __host__ __device__ Quaternion inverse() const {
      fl nsq = a * a + b * b + c * c + d * d;
      return Quaternion(a / nsq, -b / nsq, -c / nsq, -d / nsq);
    }
};

} /* namespace libmolgrid */

#endif /* QUATERNION_H_ */
