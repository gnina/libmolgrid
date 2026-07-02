/*
 * grid_interpolater_gpu.mm
 *
 * Metal (MPS) GPU dispatch for GridInterpolater::forward (GPU overloads).
 * Replaces the original grid_interpolater.cu.
 *
 * The CPU implementation (isCUDA=false overloads) remains in grid_interpolater.cpp.
 */

#include "libmolgrid/grid_interpolater.h"
#include "libmolgrid/metal_context.h"

namespace libmolgrid {

template <typename Dtype>
void GridInterpolater::forward(Vec3 in_center,
                                const Grid<Dtype, 4, true>& in,
                                const Transform& transform,
                                Vec3 out_center,
                                Grid<Dtype, 4, true>& out) const
{
    checkGrids(in, out);
    Vec3 center    = transform.get_rotation_center();
    float  in_radius = in_dimension  / 2.0f;
    float  out_radius= out_dimension / 2.0f;
    Vec3 in_origin = {in_center.x  - in_radius,
                        in_center.y  - in_radius,
                        in_center.z  - in_radius};
    Vec3 out_origin= {out_center.x - out_radius,
                        out_center.y - out_radius,
                        out_center.z - out_radius};

    Quaternion invQ  = transform.get_quaternion().inverse();
    Vec3 t         = transform.get_translation();
    Vec3 untranslate = {-t.x - center.x,
                          -t.y - center.y,
                          -t.z - center.z};

    QuatParams qp { invQ.R_component_1(), invQ.R_component_2(),
                    invQ.R_component_3(), invQ.R_component_4() };

    unsigned K = in.dimension(0);
    for (unsigned c = 0; c < K; c++) {
        // Grid<Dtype,3,true> is just a view into the same buffer.
        const float* in_ch  = reinterpret_cast<const float*>(in[c].data());
        float*       out_ch = reinterpret_cast<float*>(out[c].data());

        MetalContext::instance().grid_interpolate(
            in_ch,  in_dim,  in_origin,  in_resolution,
            qp, untranslate, center,
            out_ch, out_dim, out_origin, out_resolution);
    }
}

template void GridInterpolater::forward(Vec3,
    const Grid<float, 4, true>&, const Transform&,
    Vec3, Grid<float, 4, true>&) const;

} // namespace libmolgrid
