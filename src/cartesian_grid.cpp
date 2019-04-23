/*
 * cartesian_grid.cpp
 *
 *  Created on: Apr 19, 2019
 *      Author: dkoes
 */

#include <libmolgrid/cartesian_grid.h>

namespace libmolgrid {

template class CartesianGrid< Grid<float, 3, false> >;
template class CartesianGrid< Grid<float, 3, true> >;
template class CartesianGrid< Grid<double, 3, false> >;
template class CartesianGrid< Grid<double, 3, true> >;

template class CartesianGrid< ManagedGrid<float, 3> >;
template class CartesianGrid< ManagedGrid<double, 3> >;

} /* namespace libmolgrid */
