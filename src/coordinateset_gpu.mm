/*
 * coordinateset_gpu.mm
 *
 * Metal (MPS) GPU dispatch for CoordinateSet type-summation.
 * Replaces the original coordinateset.cu.
 */

#include "libmolgrid/coordinateset.h"
#include "libmolgrid/metal_context.h"

namespace libmolgrid {

void CoordinateSet::sum_types(Grid<float, 1, true>& sum, bool zerofirst) const
{
    if (zerofirst) sum.fill_zero();
    int NT = num_types();

    if (!has_vector_types()) {
        MetalContext::instance().sum_index_types(
            type_index.gpu().data(),
            type_index.size(),
            NT,
            sum.data());
    } else {
        MetalContext::instance().sum_vector_types(
            type_vector.gpu().data(),
            type_vector.dimension(0),
            NT,
            sum.data());
    }
}

} // namespace libmolgrid
