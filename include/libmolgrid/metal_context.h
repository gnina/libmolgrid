/*
 * metal_context.h
 *
 * Singleton that owns the Metal device, command queue, compiled shader library,
 * and per-kernel compute pipeline states.  All GPU dispatch goes through here.
 *
 * The API is plain C++ so that callers (.cpp / .mm source files) can use it
 * without pulling in any Objective-C headers.  The implementation is in
 * metal_context.mm (Objective-C++).
 */

#ifndef METAL_CONTEXT_H_
#define METAL_CONTEXT_H_

#include <cstddef>
#include "libmolgrid/common.h"

namespace libmolgrid {

// ---- POD structs passed to Metal kernels as constant buffers ---------------

// Mirrors GridMaker member variables needed inside the Metal shaders.
struct GridMakerParams {
    float resolution;
    float dimension;
    float radius_scale;
    float gaussian_radius_multiple;
    float final_radius_multiple;
    float A, B, C;   // Gaussian density coefficients
    float D, E;       // Backward-pass coefficients
    unsigned dim;     // grid points per side
    int binary;       // bool as int for Metal alignment
    int radii_type_indexed;
};

// Quaternion POD for Metal kernels
struct QuatParams {
    float a, b, c, d;
};

// Grid layout descriptor (pointer + up to 5 dimensions + strides)
struct GridDesc {
    unsigned long long ptr; // cast from void* (kernel receives device pointer via buffer slot)
    unsigned dim[5];
    unsigned off[5];
    unsigned ndim;
};

// ---- MetalContext -----------------------------------------------------------

class MetalContext {
public:
    /// Return the process-wide singleton.
    static MetalContext& instance();

    // ---- Transform kernels --------------------------------------------------

    void transform_forward(unsigned n,
                           const QuatParams& Q,
                           Vec3 center, Vec3 translate,
                           const float* in, float* out,
                           bool dotranslate);

    void transform_backward(unsigned n,
                            const QuatParams& invQ,
                            Vec3 center, Vec3 translate,
                            const float* in, float* out,
                            bool dotranslate);

    // ---- CoordinateSet kernels ----------------------------------------------

    void sum_vector_types(const float* types, unsigned natoms, unsigned ntypes,
                          float* sum);

    void sum_index_types(const float* type_index, unsigned natoms,
                         unsigned ntypes, float* sum);

    // ---- GridMaker forward kernels ------------------------------------------

    // Index-typed atoms (binary or gaussian)
    void forward_index(const GridMakerParams& gm,
                       Vec3 grid_origin,
                       const float* coords, unsigned natoms,
                       const float* type_index,
                       const float* radii,
                       float* out,
                       unsigned ntypes, unsigned dim,
                       bool binary);

    // Vector-typed atoms (binary or gaussian, optionally type-indexed radii)
    void forward_vec(const GridMakerParams& gm,
                     Vec3 grid_origin,
                     const float* coords, unsigned natoms,
                     const float* type_vector, unsigned ntypes,
                     const float* radii, float maxradius,
                     float* out, unsigned dim,
                     bool binary, bool radii_type_indexed);

    // ---- GridMaker backward kernels -----------------------------------------

    void backward_index(const GridMakerParams& gm,
                        Vec3 grid_origin,
                        const float* coords,
                        const float* type_index,
                        const float* radii,
                        const float* grid,
                        float* atom_gradients,
                        unsigned natoms, unsigned ntypes, unsigned dim);

    void backward_vec(const GridMakerParams& gm,
                      Vec3 grid_origin,
                      const float* coords,
                      const float* type_vector, unsigned ntypes,
                      const float* radii,
                      const float* grid,
                      float* atom_gradients,
                      float* type_gradients,
                      unsigned natoms, unsigned dim,
                      bool radii_type_indexed);

    void backward_gradients(const GridMakerParams& gm,
                            Vec3 grid_origin,
                            const float* coords,
                            const float* type_vector, unsigned ntypes,
                            const float* radii,
                            const float* diff,
                            const float* atom_gradients,
                            const float* type_gradients,
                            float* diffdiff,
                            float* atom_diffdiff,
                            float* type_diffdiff,
                            unsigned natoms, unsigned dim,
                            bool radii_type_indexed);

    void backward_relevance(const GridMakerParams& gm,
                            Vec3 grid_origin,
                            const float* coords,
                            const float* type_index,
                            const float* radii,
                            const float* density,
                            const float* diff,
                            float* relevance,
                            unsigned natoms, unsigned ntypes, unsigned dim);

    // ---- GridInterpolater kernel --------------------------------------------

    void grid_interpolate(const float* in,  unsigned in_dim,
                          Vec3 in_origin,  float in_res,
                          const QuatParams& invQ,
                          Vec3 untranslate, Vec3 center,
                          float* out, unsigned out_dim,
                          Vec3 out_origin, float out_res);

    // ---- Utility ------------------------------------------------------------

    /// Block the CPU until all enqueued GPU work completes.
    void synchronize();

    /// Find the maximum value in a float array on GPU (uses CPU on unified mem).
    float max_element(const float* data, size_t n);

private:
    MetalContext();
    ~MetalContext();
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    struct Impl;
    Impl* impl_;
};

} // namespace libmolgrid

#endif /* METAL_CONTEXT_H_ */
