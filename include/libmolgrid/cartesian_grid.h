/** \file cartesian_grid.h
 *
 *  Wrapper around grid object that imposes Cartesian coordinates
 */

#ifndef CARTESIAN_GRID_H_
#define CARTESIAN_GRID_H_

#include "libmolgrid/grid.h"
#include "libmolgrid/managed_grid.h"

namespace libmolgrid {

// Docstring_CartesianGrid
/** \brief Wrapper around grid of type G that imposes Cartesian coordinates.
 * Includes center and resolution and supports (eventually) interpolation.
 */
template <class G>
class CartesianGrid {
    G grid_;
    float3 center_ = {0,};
    float resolution_ = 0.0;
  public:
    /// Initialize CartesianGrid
    CartesianGrid(const G& g, float3 c, float res): grid_(g), center_(c), resolution_(res) {}
    ~CartesianGrid() {}

    /// return center of grid
    float3 center() const { return center_; }
    /// return resolution of grid
    float resolution() const { return resolution_; }
    /// return underlying grid
    G& grid() { return grid_; }
    const G& grid() const { return grid_; }

    /// return linear interpolation of value at specify position
    typename G::type interpolate(float x, float y, float z) const;

};

using CartesianMGrid = CartesianGrid<ManagedGrid<float, 3> >;

} /* namespace libmolgrid */

#endif /* CARTESIAN_GRID_H_ */
