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

    /* \brief Generate grid tensor from CPU atomic data.  Grid must be properly sized.
     * @param[in] center of grid
     * @param[in] coordinate set
     * @param[out] a 4D grid
     */
    template <typename Dtype>
    void forward(float3 grid_center, const CoordinateSet& in, Grid<Dtype, 4, false>& out) const {
      if(in.has_indexed_types()) {
        forward(grid_center, in.coord.cpu(), in.type_index.cpu(), in.radius.cpu(), out);
      } else {
        throw std::invalid_argument("Type vector gridding not implemented yet");
      }
    }

    /* \brief Generate grid tensor from GPU atomic data.  Grid must be properly sized.
     * @param[in] center of grid
     * @param[in] coordinate set
     * @param[out] a 4D grid
     */
    template <typename Dtype>
    void forward(float3 grid_center, const CoordinateSet& in, Grid<Dtype, 4, true>& out) const {
      if(in.has_indexed_types()) {
        forward(grid_center, in.coord.gpu(), in.type_index.gpu(), in.radius.gpu(), out);
      } else {
        throw std::invalid_argument("Type vector gridding not implemented yet");
      }
    }

    /* \brief Generate grid tensort from CPU atomic data.  Grid must be properly sized.
     * @param[in] center of grid
     * @param[in] coordinates (Nx3)
     * @param[in] type indices (N integers stored as floats)
     * @param[in] radii (N)
     * @param[out] a 4D grid
     */
    template <typename Dtype>
    void forward(float3 grid_center, const Grid<float, 2, false>& coords,
        const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
        Grid<Dtype, 4, false>& out) const;

    /* \brief Generate grid tensort from GPU atomic data.  Grid must be properly sized.
     * @param[in] center of grid
     * @param[in] coordinates (Nx3)
     * @param[in] type indices (N integers stored as floats)
     * @param[in] radii (N)
     * @param[out] a 4D gri
     */
    template <typename Dtype>
    __host__ void forward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
        Grid<Dtype, 4, false>& out) const;

};

} /* namespace libmolgrid */

#endif /* GRID_MAKER_H_ */
