/*
 * grid_maker.h
 *
 *  Grid generation form atomic data.
 *  Created on: Mar 26, 2019
 *      Author: dkoes
 */

#ifndef GRID_MAKER_H_
#define GRID_MAKER_H_

#include "coordinateset.h"
#include "grid.h"

namespace libmolgrid {

class GridMaker {
  public:

    // need to implement all options
    GridMaker();
    virtual ~GridMaker() {}


    //need function that returns grid dimensions (tuple?)

    /* \brief Generate grid tensort from CPU atomic data.  Grid must be properly sized.
     * @param[in] center of grid
     * @param[in] coordinate set
     * @param[out] a 4D grid
     */
    template <typename Dtype>
    void forward(float3 grid_center, const CoordinateSet& in, Grid<Dtype, 4, false>& out) const;

    /* \brief Generate grid tensort from GPU atomic data.  Grid must be properly sized.
     * @param[in] center of grid
     * @param[in] coordinate set
     * @param[out] a 4D grid
     */
    template <typename Dtype>
    __host__ void forward(float3 grid_center, const CoordinateSet& in, Grid<Dtype, 4, true>& out) const;


};

} /* namespace libmolgrid */

#endif /* GRID_MAKER_H_ */
