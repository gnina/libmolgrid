/*
 * transform_gpu.mm
 *
 * Metal (MPS) GPU dispatch for Transform class.
 * Replaces the original transform.cu.
 */

#include "libmolgrid/transform.h"
#include "libmolgrid/metal_context.h"

namespace libmolgrid {

template <typename Dtype>
void Transform::forward(const Grid<Dtype, 2, true>& in,
                         Grid<Dtype, 2, true>& out,
                         bool dotranslate) const
{
    checkGrids(in, out);
    unsigned N = in.dimension(0);
    if (N == 0) return;

    QuatParams qp { Q.R_component_1(), Q.R_component_2(),
                    Q.R_component_3(), Q.R_component_4() };

    MetalContext::instance().transform_forward(
        N, qp, center, translate,
        reinterpret_cast<const float*>(in.data()),
        reinterpret_cast<float*>(out.data()),
        dotranslate);
}

template void Transform::forward(const Grid<float, 2, true>&,
                                  Grid<float, 2, true>&, bool) const;

template <typename Dtype>
void Transform::backward(const Grid<Dtype, 2, true>& in,
                          Grid<Dtype, 2, true>& out,
                          bool dotranslate) const
{
    checkGrids(in, out);
    unsigned N = in.dimension(0);
    if (N == 0) return;

    Quaternion invQ = Q.inverse();
    QuatParams qp { invQ.R_component_1(), invQ.R_component_2(),
                    invQ.R_component_3(), invQ.R_component_4() };

    MetalContext::instance().transform_backward(
        N, qp, center, translate,
        reinterpret_cast<const float*>(in.data()),
        reinterpret_cast<float*>(out.data()),
        dotranslate);
}

template void Transform::backward(const Grid<float,  2, true>&,
                                   Grid<float,  2, true>&, bool) const;
template void Transform::backward(const Grid<double, 2, true>&,
                                   Grid<double, 2, true>&, bool) const;

} // namespace libmolgrid
