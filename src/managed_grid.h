/*
 * managed_grid.h
 *
 * A grid that manages its own memory using a shared pointer.
 *
 */

#ifndef MANAGED_GRID_H_
#define MANAGED_GRID_H_

#include<memory>
#include <grid.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

namespace libmolgrid {

/** \brief A dense grid whose memory is managed by the class.
 *
 * Memory is allocated as unified memory so it can be safely accessed
 * from either the CPU or GPU.
 */
template<typename Dtype, std::size_t NumDims>
class ManagedGrid : public Grid<Dtype, NumDims, true> {
  protected:
    std::shared_ptr<Dtype> ptr; //shared pointer lets us not worry about copying the grid

  public:
    template<typename... I>
    ManagedGrid(I... sizes): Grid<Dtype, NumDims, true>(nullptr, sizes...) {
      //allocate buffer
      size_t sz = 1;
      for(unsigned i = 0; i < NumDims; i++) {
        sz *= this->dims[i];
      }
      cudaMallocManaged((void**)&this->buffer,sz*sizeof(Dtype));
      memset(this->buffer, 0, sz*sizeof(Dtype));
      ptr = std::shared_ptr<Dtype>(this->buffer,cudaFree);
    }
};

#define EXPAND_MGRID_DEFINITIONS(SIZE) \
typedef ManagedGrid<float, SIZE> MGrid##SIZE##f; \
typedef ManagedGrid<double, SIZE> MGrid##SIZE##d;


EXPAND_MGRID_DEFINITIONS(1)
EXPAND_MGRID_DEFINITIONS(2)
EXPAND_MGRID_DEFINITIONS(3)
EXPAND_MGRID_DEFINITIONS(4)
EXPAND_MGRID_DEFINITIONS(5)
EXPAND_MGRID_DEFINITIONS(6)

}
#endif /* MANAGED_GRID_H_ */
