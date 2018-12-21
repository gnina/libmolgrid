/** \file grid.h
 *
 *  Wrapper classes for holding dense, multi-dimensional arrays of data.
 */

#ifndef GRID_H_
#define GRID_H_

#include <boost/multi_array.hpp>
#include <cassert>
#include "common.h"
#include <iostream>
namespace libmolgrid {

template<typename Dt, std::size_t ND> class ManagedGrid; //predeclare for constructor

/**
 * \class Grid
 * A dense array of memory stored on the CPU.  The memory is owned
 * and managed external to this class.  The location and size of
 * the memory should not change during the lifetime of the grid.
 * If isCUDA is true, data should only be accessed in kernels.
 */
template<typename Dtype, std::size_t NumDims, bool isCUDA = false>
class Grid {
  protected:
    //these should be read only, but I need to set them in the constructor
    //outside an initializer list, so hide them
    Dtype * buffer; /// raw pointer to data
    size_t dims[NumDims];
    size_t offs[NumDims];

  public:
    /// dimensions along each axis
    CUDA_CALLABLE_MEMBER inline const size_t * dimensions() const { return dims; }
    /// dimensions along specified axis
    CUDA_CALLABLE_MEMBER inline size_t dimension(size_t i) const { assert(i<NumDims); return dims[i]; }

    /// offset for each dimension, all indexing calculations use this
    CUDA_CALLABLE_MEMBER inline const size_t * offsets() const { return offs; }
    /// offset for each dimension, all indexing calculations use this
    CUDA_CALLABLE_MEMBER inline size_t offset(size_t i) const { return offs[i]; }

    /// number of elements in grid
    CUDA_CALLABLE_MEMBER inline size_t size() const {
      size_t ret = 1;
      for(unsigned i = 0; i < NumDims; i++) {
        ret *= dims[i];
      }
      return ret;
    }

    /// pointer to underlying data
    CUDA_CALLABLE_MEMBER inline Dtype * data() const { return buffer; }

    /** \brief Grid constructor
     *
     * Provide pointer and dimensions specified as arguments
    */
    template<typename... I>
    Grid(Dtype *const d, I... sizes):
      buffer(d), dims{  static_cast<size_t>(sizes)...} {
      static_assert(NumDims == sizeof...(sizes),"Incorrect number of grid dimensions");
      offs[NumDims-1] = 1;
      #pragma unroll
      for(int i = NumDims-1; i > 0; i--) {
        offs[i-1] = dims[i]*offs[i];
      }
    }

    /** \brief Construct Grid from ManagedGrid
     *
     * The managed grid retains responsibility for the memory.
     * This constructor ensures that unified memory is prefetched
     * to correct device/host.
     */
    Grid(const ManagedGrid<Dtype, NumDims>& mg); //implementation is in managed_grid.h

    Grid(const Grid&) = default;
    ~Grid() = default;

    /** \brief Bracket indexing.
     *
     *  Accessing data this way will be safe (indices are checked) and convenient,
     *  but not maximally efficient (unless the compiler is really good).
     *  Use operator() for fastest (but unchecked) access or access data directly.
     */
    CUDA_CALLABLE_MEMBER Grid<Dtype,NumDims-1,isCUDA> operator[](size_t i) {
      assert(i < dims[0]);
      return Grid<Dtype,NumDims-1,isCUDA>(*this, i);
    }

    /** \brief Initializer list indexing
     *
     */
    template<typename... I>
    CUDA_CALLABLE_MEMBER inline Dtype& operator()(I... indices) {
      static_assert(NumDims == sizeof...(indices),"Incorrect number of grid indices");

      size_t idx[NumDims] = { static_cast<size_t>(indices)...};
      size_t pos = 0;
      #pragma unroll
      for(unsigned i = 0; i < NumDims; i++) { //surely the compiler will unroll this...
        pos += idx[i]*offs[i];
      }
      return buffer[pos];
    }

    // constructor used by operator[]
    CUDA_CALLABLE_MEMBER
    explicit Grid(const Grid<Dtype,NumDims+1,isCUDA>& G, size_t i): buffer(&G.data()[i*G.offset(0)]) {
      //slice off first dimension
      for(size_t i = 0; i < NumDims; i++) {
        dims[i] = G.dimension(i+1);
        offs[i] = G.offset(i+1);
      }
    }

};

// class specialization of grid to make final operator[] return scalar
// Note: if we start adding more functionality to Grid, we will need
// to refactor this into a separate GridAccessor class
template<typename Dtype, bool isCUDA >
class Grid<Dtype,1,isCUDA> {
  protected:
    Dtype * buffer;
    size_t dims[1]; /// length of array

  public:
    /// dimensions along each axis
    CUDA_CALLABLE_MEMBER inline const size_t * dimensions() const { return dims; }
    /// dimensions along specified axis
    CUDA_CALLABLE_MEMBER inline size_t dimension(size_t i) const { assert(i<1); return dims[i]; }

    /// number of elements in grid
    CUDA_CALLABLE_MEMBER inline size_t size() const { return dims[0]; }

    /// pointer to underlying data
    CUDA_CALLABLE_MEMBER inline Dtype * data() const { return buffer; }

    Grid(Dtype* const d, size_t sz):
      buffer(d), dims{sz} { }

    Grid(const ManagedGrid<Dtype, 1>& mg); //implementation is in managed_grid.h


    CUDA_CALLABLE_MEMBER inline Dtype& operator[](size_t i) {
      assert(i < dims[0]);
      return buffer[i];
    }

    CUDA_CALLABLE_MEMBER inline Dtype& operator()(size_t a) {
      return buffer[a];
    }

    //only called from regular Grid
    CUDA_CALLABLE_MEMBER
    explicit Grid<Dtype,1,isCUDA>(const Grid<Dtype,2,isCUDA>& G, size_t i):
      buffer(&G.data()[i*G.offset(0)]), dims{G.dimension(1)} {}

};

#define EXPAND_GRID_DEFINITIONS(SIZE) \
typedef Grid<float, SIZE, false> Grid##SIZE##f; \
typedef Grid<double, SIZE, false> Grid##SIZE##d; \
typedef Grid<float, SIZE, true> Grid##SIZE##fCUDA; \
typedef Grid<double, SIZE, true> Grid##SIZE##dCUDA;


EXPAND_GRID_DEFINITIONS(1)
EXPAND_GRID_DEFINITIONS(2)
EXPAND_GRID_DEFINITIONS(3)
EXPAND_GRID_DEFINITIONS(4)
EXPAND_GRID_DEFINITIONS(5)
EXPAND_GRID_DEFINITIONS(6)

}

#endif /* GRID_H_ */
