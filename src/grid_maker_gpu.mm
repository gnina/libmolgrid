/*
 * grid_maker_gpu.mm
 *
 * Metal (MPS) GPU dispatch for GridMaker forward / backward passes.
 * Replaces the original grid_maker.cu.
 *
 * The CPU implementations remain in grid_maker.cpp.
 * This file only provides the isCUDA=true overloads.
 */

#include "libmolgrid/grid_maker.h"
#include "libmolgrid/metal_context.h"
#include <algorithm>  // std::max_element

namespace libmolgrid {

// ---------------------------------------------------------------------------
// Helper: access GridMaker protected members via a derived-struct accessor.
// In C++ a derived class may access protected members of its own instances.
// ---------------------------------------------------------------------------
struct GridMakerAccessor : public GridMaker {
    static GridMakerParams build(const GridMakerAccessor& g) {
        GridMakerParams p;
        p.resolution               = g.resolution;
        p.dimension                = g.dimension;
        p.radius_scale             = g.radius_scale;
        p.gaussian_radius_multiple = g.gaussian_radius_multiple;
        p.final_radius_multiple    = g.final_radius_multiple;
        p.A  = g.A;  p.B  = g.B;  p.C  = g.C;
        p.D  = g.D;  p.E  = g.E;
        p.dim                      = g.dim;
        p.binary                   = g.binary ? 1 : 0;
        p.radii_type_indexed       = g.radii_type_indexed ? 1 : 0;
        return p;
    }
};

static GridMakerParams makeParams(const GridMaker& gm) {
    return GridMakerAccessor::build(
        static_cast<const GridMakerAccessor&>(gm));
}

// ===========================================================================
// Forward  –  Index types
// ===========================================================================

// Helper: run index-type forward into a float buffer, then copy to Dtype if needed.
static void forward_index_impl(const GridMaker& gm, Vec3 grid_origin,
                                const Grid<float,2,true>& coords,
                                const Grid<float,1,true>& type_index,
                                const Grid<float,1,true>& radii,
                                float* out_ptr, unsigned ntypes, unsigned dim, bool binary)
{
    MetalContext::instance().forward_index(
        makeParams(gm), grid_origin,
        coords.data(),    coords.dimension(0),
        type_index.data(),
        radii.data(),
        out_ptr,
        ntypes, dim, binary);
}

template <typename Dtype>
void GridMaker::forward(Vec3 grid_center,
                         const Grid<float, 2, true>& coords,
                         const Grid<float, 1, true>& type_index,
                         const Grid<float, 1, true>& radii,
                         Grid<Dtype, 4, true>& out) const
{
    check_index_args(coords, type_index, radii, out);
    if (radii_type_indexed)
        throw std::invalid_argument(
            "Type indexed radii not supported with index types.");
    if (dim == 0)
        throw std::invalid_argument("Zero sized grid.");

    Vec3 grid_origin = get_grid_origin(grid_center);
    out.fill_zero();
    if (coords.dimension(0) == 0) return;

    if constexpr (std::is_same<Dtype, float>::value) {
        forward_index_impl(*this, grid_origin, coords, type_index, radii,
                           out.data(), out.dimension(0), dim, binary);
    } else {
        // Metal GPU only supports float; compute into a temp buffer then widen.
        size_t n = out.size();
        std::vector<float> tmp(n, 0.0f);
        forward_index_impl(*this, grid_origin, coords, type_index, radii,
                           tmp.data(), out.dimension(0), dim, binary);
        for (size_t i = 0; i < n; ++i) out.data()[i] = static_cast<Dtype>(tmp[i]);
    }
}

template void GridMaker::forward(Vec3, const Grid<float,2,true>&,
    const Grid<float,1,true>&, const Grid<float,1,true>&, Grid<float,4,true>&) const;
template void GridMaker::forward(Vec3, const Grid<float,2,true>&,
    const Grid<float,1,true>&, const Grid<float,1,true>&, Grid<double,4,true>&) const;

// Batched overloads delegate to the single-example overload (loop on CPU).
template void GridMaker::forward<float,2,true>(const Grid<float,2,true>&,
    const Grid<float,3,true>&, const Grid<float,2,true>&,
    const Grid<float,2,true>&, Grid<float,5,true>&) const;
template void GridMaker::forward<float,3,true>(const Grid<float,2,true>&,
    const Grid<float,3,true>&, const Grid<float,3,true>&,
    const Grid<float,2,true>&, Grid<float,5,true>&) const;
template void GridMaker::forward<double,2,true>(const Grid<float,2,true>&,
    const Grid<float,3,true>&, const Grid<float,2,true>&,
    const Grid<float,2,true>&, Grid<double,5,true>&) const;
template void GridMaker::forward<double,3,true>(const Grid<float,2,true>&,
    const Grid<float,3,true>&, const Grid<float,3,true>&,
    const Grid<float,2,true>&, Grid<double,5,true>&) const;

// ===========================================================================
// Forward  –  Vector types
// ===========================================================================

static void forward_vec_impl(const GridMaker& gm, Vec3 grid_origin,
                              const Grid<float,2,true>& coords,
                              const Grid<float,2,true>& type_vector,
                              const Grid<float,1,true>& radii, float maxr,
                              float* out_ptr, unsigned dim, bool binary, bool rti)
{
    unsigned ntypes = type_vector.dimension(1);
    MetalContext::instance().forward_vec(
        makeParams(gm), grid_origin,
        coords.data(),      coords.dimension(0),
        type_vector.data(), ntypes,
        radii.data(),       maxr,
        out_ptr,            dim,
        binary,             rti);
}

template <typename Dtype>
void GridMaker::forward(Vec3 grid_center,
                         const Grid<float, 2, true>& coords,
                         const Grid<float, 2, true>& type_vector,
                         const Grid<float, 1, true>& radii,
                         Grid<Dtype, 4, true>& out) const
{
    check_vector_args(coords, type_vector, radii, out);
    Vec3 grid_origin = get_grid_origin(grid_center);

    out.fill_zero();
    if (coords.dimension(0) == 0) return;

    float maxr = 0.0f;
    if (radii_type_indexed) {
        const float* rd = radii.data();
        maxr = *std::max_element(rd, rd + radii.size());
    }

    if constexpr (std::is_same<Dtype, float>::value) {
        forward_vec_impl(*this, grid_origin, coords, type_vector, radii, maxr,
                         out.data(), dim, binary, radii_type_indexed);
    } else {
        size_t n = out.size();
        std::vector<float> tmp(n, 0.0f);
        forward_vec_impl(*this, grid_origin, coords, type_vector, radii, maxr,
                         tmp.data(), dim, binary, radii_type_indexed);
        for (size_t i = 0; i < n; ++i) out.data()[i] = static_cast<Dtype>(tmp[i]);
    }
}

template void GridMaker::forward(Vec3, const Grid<float,2,true>&,
    const Grid<float,2,true>&, const Grid<float,1,true>&, Grid<float,4,true>&) const;
template void GridMaker::forward(Vec3, const Grid<float,2,true>&,
    const Grid<float,2,true>&, const Grid<float,1,true>&, Grid<double,4,true>&) const;

// ===========================================================================
// Backward  –  Index types
// ===========================================================================

template <typename Dtype>
void GridMaker::backward(Vec3 grid_center,
                          const Grid<float, 2, true>& coords,
                          const Grid<float, 1, true>& type_index,
                          const Grid<float, 1, true>& radii,
                          const Grid<Dtype, 4, true>& grid,
                          Grid<Dtype, 2, true>& atom_gradients) const
{
    atom_gradients.fill_zero();
    unsigned n = coords.dimension(0);
    if (n != type_index.size())
        throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
    if (n != radii.size())
        throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");
    if (n != atom_gradients.dimension(0))
        throw std::invalid_argument("Gradient dimension doesn't equal number of coordinates");
    if (coords.dimension(1) != 3)
        throw std::invalid_argument("Coordinates wrong secondary dimension (!= 3)");
    if (radii_type_indexed)
        throw std::invalid_argument("Type indexed radii not supported with index types.");

    if (n == 0) return;
    Vec3 grid_origin = get_grid_origin(grid_center);

    MetalContext::instance().backward_index(
        makeParams(*this), grid_origin,
        coords.data(), type_index.data(), radii.data(),
        reinterpret_cast<const float*>(grid.data()),
        reinterpret_cast<float*>(atom_gradients.data()),
        n, grid.dimension(0), dim);
}

template void GridMaker::backward(Vec3, const Grid<float,2,true>&,
    const Grid<float,1,true>&, const Grid<float,1,true>&,
    const Grid<float,4,true>&, Grid<float,2,true>&) const;
template void GridMaker::backward(Vec3, const Grid<float,2,true>&,
    const Grid<float,1,true>&, const Grid<float,1,true>&,
    const Grid<double,4,true>&, Grid<double,2,true>&) const;

// ===========================================================================
// Backward  –  Vector types
// ===========================================================================

template <typename Dtype>
void GridMaker::backward(Vec3 grid_center,
                          const Grid<float, 2, true>& coords,
                          const Grid<float, 2, true>& type_vector,
                          const Grid<float, 1, true>& radii,
                          const Grid<Dtype, 4, true>& grid,
                          Grid<Dtype, 2, true>& atom_gradients,
                          Grid<Dtype, 2, true>& type_gradients) const
{
    atom_gradients.fill_zero();
    type_gradients.fill_zero();
    unsigned n      = coords.dimension(0);
    unsigned ntypes = type_vector.dimension(1);

    if (n != type_vector.dimension(0))
        throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
    if (ntypes != grid.dimension(0))
        throw std::invalid_argument("Channels in diff doesn't equal number of types");
    if (n != atom_gradients.dimension(0))
        throw std::invalid_argument("Atom gradient dimension doesn't equal number of coordinates");
    if (n != type_gradients.dimension(0))
        throw std::invalid_argument("Type gradient dimension doesn't equal number of coordinates");
    if (type_gradients.dimension(1) != ntypes)
        throw std::invalid_argument("Type gradient dimension has wrong number of types");
    if (coords.dimension(1) != 3)
        throw std::invalid_argument("Need x,y,z for coord");

    if (radii_type_indexed) {
        if (ntypes != radii.size())
            throw std::invalid_argument("Radii dimension doesn't equal number of types");
    } else {
        if (n != radii.size())
            throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");
    }

    if (n == 0) return;
    Vec3 grid_origin = get_grid_origin(grid_center);

    MetalContext::instance().backward_vec(
        makeParams(*this), grid_origin,
        coords.data(), type_vector.data(), ntypes, radii.data(),
        reinterpret_cast<const float*>(grid.data()),
        reinterpret_cast<float*>(atom_gradients.data()),
        reinterpret_cast<float*>(type_gradients.data()),
        n, dim, radii_type_indexed);
}

template void GridMaker::backward(Vec3, const Grid<float,2,true>&,
    const Grid<float,2,true>&, const Grid<float,1,true>&,
    const Grid<float,4,true>&, Grid<float,2,true>&, Grid<float,2,true>&) const;

// ===========================================================================
// backward_gradients
// ===========================================================================

template <typename Dtype>
void GridMaker::backward_gradients(Vec3 grid_center,
                                    const Grid<float, 2, true>& coords,
                                    const Grid<float, 2, true>& type_vector,
                                    const Grid<float, 1, true>& radii,
                                    const Grid<Dtype, 4, true>& diff,
                                    const Grid<Dtype, 2, true>& atom_gradients,
                                    const Grid<Dtype, 2, true>& type_gradients,
                                    Grid<Dtype, 4, true>& diffdiff,
                                    Grid<Dtype, 2, true>& atom_diffdiff,
                                    Grid<Dtype, 2, true>& type_diffdiff)
{
    unsigned n      = coords.dimension(0);
    unsigned ntypes = type_vector.dimension(1);
    check_vector_args(coords, type_vector, radii, diff);

    if (binary) throw std::invalid_argument("Binary densities not supported");

    atom_diffdiff.fill_zero();
    type_diffdiff.fill_zero();
    diffdiff.fill_zero();

    if (n == 0) return;
    Vec3 grid_origin = get_grid_origin(grid_center);

    MetalContext::instance().backward_gradients(
        makeParams(*this), grid_origin,
        coords.data(), type_vector.data(), ntypes, radii.data(),
        reinterpret_cast<const float*>(diff.data()),
        reinterpret_cast<const float*>(atom_gradients.data()),
        reinterpret_cast<const float*>(type_gradients.data()),
        reinterpret_cast<float*>(diffdiff.data()),
        reinterpret_cast<float*>(atom_diffdiff.data()),
        reinterpret_cast<float*>(type_diffdiff.data()),
        n, dim, radii_type_indexed);
}

template void GridMaker::backward_gradients(Vec3,
    const Grid<float,2,true>&, const Grid<float,2,true>&,
    const Grid<float,1,true>&, const Grid<float,4,true>&,
    const Grid<float,2,true>&, const Grid<float,2,true>&,
    Grid<float,4,true>&, Grid<float,2,true>&, Grid<float,2,true>&);

// ===========================================================================
// backward_relevance
// ===========================================================================

template <typename Dtype>
void GridMaker::backward_relevance(Vec3 grid_center,
                                    const Grid<float, 2, true>& coords,
                                    const Grid<float, 1, true>& type_index,
                                    const Grid<float, 1, true>& radii,
                                    const Grid<Dtype, 4, true>& density,
                                    const Grid<Dtype, 4, true>& diff,
                                    Grid<Dtype, 1, true>& relevance) const
{
    relevance.fill_zero();
    unsigned n = coords.dimension(0);
    if (n != type_index.size())
        throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
    if (n != relevance.size())
        throw std::invalid_argument("Relevance dimension doesn't equal number of coordinates");
    if (n != radii.size())
        throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");
    if (n == 0) return;

    Vec3 grid_origin = get_grid_origin(grid_center);

    MetalContext::instance().backward_relevance(
        makeParams(*this), grid_origin,
        coords.data(), type_index.data(), radii.data(),
        reinterpret_cast<const float*>(density.data()),
        reinterpret_cast<const float*>(diff.data()),
        reinterpret_cast<float*>(relevance.data()),
        n, density.dimension(0), dim);
}

template void GridMaker::backward_relevance(Vec3,
    const Grid<float,2,true>&, const Grid<float,1,true>&,
    const Grid<float,1,true>&, const Grid<float,4,true>&,
    const Grid<float,4,true>&, Grid<float,1,true>&) const;

} // namespace libmolgrid
