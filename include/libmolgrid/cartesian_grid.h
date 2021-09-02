/** \file cartesian_grid.h
 *
 *  Wrapper around grid object that imposes Cartesian coordinates
 */

#ifndef CARTESIAN_GRID_H_
#define CARTESIAN_GRID_H_

#include <type_traits>
#include "libmolgrid/grid.h"
#include "libmolgrid/managed_grid.h"

namespace libmolgrid {

// Docstring_CartesianGrid
/** \brief Wrapper around grid of type G that imposes Cartesian coordinates.
 * Includes center and resolution and supports (eventually) interpolation.
 *
 */
template <class G>
class CartesianGrid {
    G grid_;
    size_t dimensions[3] = {0,}; //number of grid points
    float3 center_ = {0,};
    float3 origin_ = {0,};
    float resolution_ = 0.0;
  public:
    /// Initialize CartesianGrid
    CartesianGrid(const G& g, float3 c, float res): grid_(g), center_(c), resolution_(res) {
    	dimensions[0] = g.dimension(G::N-3);
    	dimensions[1] = g.dimension(G::N-2);
    	dimensions[2] = g.dimension(G::N-1);
    	origin_.x = center_.x-dimensions[0]*resolution_/2.0;
    	origin_.y = center_.y-dimensions[1]*resolution_/2.0;
    	origin_.z = center_.z-dimensions[2]*resolution_/2.0;

    }
    ~CartesianGrid() {}

    /// return center of grid
    float3 center() const { return center_; }
    /// return resolution of grid
    float resolution() const { return resolution_; }
    /// return underlying grid
    G& grid() { return grid_; }
    const G& grid() const { return grid_; }

    /// return grid coordinates (not rounded) for Cartesian coordinates
    float3 cart2grid(float x, float y, float z) const {
    	float3 pt = { (x-origin_.x)/resolution_, (y-origin_.y)/resolution_, (z-origin_.z)/resolution_ };
    	return pt;
    }

    /// return Cartesian coordinates of provided grid position
    float3 grid2cart(unsigned i, unsigned j, unsigned k) const {
    	float3 pt = {origin_.x+i*resolution_,origin_.y+j*resolution_,origin_.z+k*resolution_};
    	return pt;
    }

    /// return linear interpolation of value at specify position
    typename G::type interpolate(size_t channel, float x, float y, float z) const;

};

using CartesianMGrid = CartesianGrid<ManagedGrid<float, 3> >;

} /* namespace libmolgrid */

#endif /* CARTESIAN_GRID_H_ */
