/** \file libmolgrid.cpp
 *  \brief Global options for libmolgrid
 */
#include "libmolgrid/libmolgrid.h"
#include "libmolgrid/config.h"
#include "libmolgrid/grid.h"
#include "libmolgrid/managed_grid.h"
#include "libmolgrid/transform.h"

namespace libmolgrid {
    std::default_random_engine random_engine;


#define INSTANTIATE_GRID_DEFINITIONS(SIZE) \
    template class Grid<float, SIZE, false>; \
    template class Grid<double, SIZE, false>; \
    template class Grid<float, SIZE, true>; \
    template class Grid<double, SIZE, true>; \
    template class ManagedGrid<float, SIZE>; \
    template class ManagedGrid<double, SIZE>;

INSTANTIATE_GRID_DEFINITIONS(1)
INSTANTIATE_GRID_DEFINITIONS(2)
INSTANTIATE_GRID_DEFINITIONS(3)
INSTANTIATE_GRID_DEFINITIONS(4)
INSTANTIATE_GRID_DEFINITIONS(5)
INSTANTIATE_GRID_DEFINITIONS(6)

}
