/*
 * cartesian_grid.cpp
 *
 *  Created on: Apr 19, 2019
 *      Author: dkoes
 */

#include <libmolgrid/cartesian_grid.h>

namespace libmolgrid {

template class CartesianGrid< Grid<float, 4, false> >;
template class CartesianGrid< Grid<float, 4, true> >;
template class CartesianGrid< Grid<double, 4, false> >;
template class CartesianGrid< Grid<double, 4, true> >;

template class CartesianGrid< ManagedGrid<float, 4> >;
template class CartesianGrid< ManagedGrid<double, 4> >;


} /* namespace libmolgrid */
