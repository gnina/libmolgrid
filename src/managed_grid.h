/*
 * managed_grid.h
 *
 * A grid that manages its own memory using a shared pointer.
 * Any libmolgrid routines that create a grid option (e.g. readers)
 * return a ManagedGrid.  Memory is allocated as unified CUDA memory.
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
      size_t sz = this->size();
      cudaMallocManaged((void**)&this->buffer,sz*sizeof(Dtype));
      memset(this->buffer, 0, sz*sizeof(Dtype));
      ptr = std::shared_ptr<Dtype>(this->buffer,cudaFree);
    }

    CUDA_CALLABLE_MEMBER const std::shared_ptr<Dtype> pointer() const { return ptr; }

    /** \brief Bracket indexing.
     *
     *  Accessing data this way will be safe (indices are checked) and convenient,
     *  but not maximally efficient (unless the compiler is really good).
     *  Use operator() for fastest (but unchecked) access or access data directly.
     */
    CUDA_CALLABLE_MEMBER ManagedGrid<Dtype,NumDims-1> operator[](size_t i) {
      assert(i < this->dims[0]);
      return ManagedGrid<Dtype,NumDims-1>(*this, i);
    }

    // constructor used by operator[]
    CUDA_CALLABLE_MEMBER
    explicit ManagedGrid(const ManagedGrid<Dtype,NumDims+1>& G, size_t i):
      Grid<Dtype, NumDims, true>(G, i), ptr(G.pointer()) {}

};

// class specialization of managed grid to make final operator[] return scalar
template<typename Dtype >
class ManagedGrid<Dtype, 1> : public Grid<Dtype, 1, true> {
  protected:
    std::shared_ptr<Dtype> ptr;

  public:

    ManagedGrid(size_t sz): Grid<Dtype, 1, true>(nullptr, sz) {
      cudaMallocManaged((void**)&this->buffer,sz*sizeof(Dtype));
      memset(this->buffer, 0, sz*sizeof(Dtype));
      ptr = std::shared_ptr<Dtype>(this->buffer,cudaFree);
    }

    //only called from regular Grid
    CUDA_CALLABLE_MEMBER
    explicit ManagedGrid<Dtype,1>(const ManagedGrid<Dtype,2>& G, size_t i):
      Grid<Dtype, 1, true>(G, i), ptr(G.pointer()) {}

};

template<typename Dtype, std::size_t NumDims, bool isCUDA>
Grid<Dtype, NumDims, isCUDA>::Grid(const ManagedGrid<Dtype, NumDims>& mg):
  buffer(mg.data()) {
  int device = -1;

  for(unsigned i = 0; i < NumDims; i++) {
    dims[i] = mg.dimension(i);
    offs[i] = mg.offset(i);
  }

  if(isCUDA) {
    //prefetch to GPU
    cudaGetDevice(&device);
  } else {
    //prefetch to host
    device = cudaCpuDeviceId;
  }
  cudaMemPrefetchAsync(this->data(),this->size(),device, NULL);
  //should I synchronize here?
}

template<typename Dtype, bool isCUDA>
Grid<Dtype, 1, isCUDA>::Grid(const ManagedGrid<Dtype, 1>& mg):
  buffer(mg.data()), dims{mg.dimension(0)} {
  int device = -1;

  if(isCUDA) {
    //prefetch to GPU
    cudaGetDevice(&device);
  } else {
    //prefetch to host
    device = cudaCpuDeviceId;
  }
  cudaMemPrefetchAsync(this->data(),this->size(),device, NULL);
  //should I synchronize here?
}

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
