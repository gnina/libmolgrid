/** \file grid.h
 *
 *  Wrapper classes for holding dense, multi-dimensional arrays of data.
 */

#ifndef GRID_H_
#define GRID_H_

#include <boost/multi_array.hpp>
#include <boost/preprocessor/repetition.hpp>
#include <boost/lexical_cast.hpp>
#include <cassert>
#include <iostream>

#include "libmolgrid/common.h"
#include "libmolgrid/libmolgrid.h"

namespace libmolgrid {

template<typename Dt, std::size_t ND> class ManagedGrid; //predeclare

inline cudaMemcpyKind copyKind(bool srcCUDA, bool destCUDA) {
  if (srcCUDA) {
    if (destCUDA)
      return cudaMemcpyDeviceToDevice;
    else
      return cudaMemcpyDeviceToHost;
  } else { //source host
    if (destCUDA)
      return cudaMemcpyHostToDevice;
    else
      return cudaMemcpyHostToHost;
  }
}

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

    template<typename... I>
    CUDA_CALLABLE_MEMBER inline size_t getPos(I... indices) const {
      static_assert(NumDims == sizeof...(indices),"Incorrect number of grid indices");

      size_t idx[NumDims] = { static_cast<size_t>(indices)...};
      size_t pos = 0;
      #pragma unroll
      for(unsigned i = 0; i < NumDims; i++) { //surely the compiler will unroll this...
        pos += idx[i]*offs[i];
      }
      return pos;
    }

    CUDA_CALLABLE_MEMBER void check_index(size_t i, size_t dim) const {
#ifndef __CUDA_ARCH__
      if(i >= dim) {
        throw std::out_of_range("Invalid range. "+itoa(i) + " >= " + itoa(dim));
      }
#else
      assert(i < dim);
#endif
    }

  public:
    using type = Dtype;
    using subgrid_t = Grid<Dtype,NumDims-1,isCUDA>;
    using managed_t = ManagedGrid<Dtype,NumDims>;
    using cpu_grid_t = Grid<Dtype,NumDims,false>;
    using gpu_grid_t = Grid<Dtype,NumDims,true>;

    static constexpr size_t N = NumDims;
    static constexpr bool GPU = isCUDA;

    /// dimensions along each axis
    CUDA_CALLABLE_MEMBER inline const size_t * dimensions() const { return dims; }
    /// dimensions along specified axis
    CUDA_CALLABLE_MEMBER inline size_t dimension(size_t i) const { check_index(i,NumDims); return dims[i]; }

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

    /// set the underlying memory buffer - use with caution!
    CUDA_CALLABLE_MEMBER inline void set_buffer(Dtype *ptr) { buffer = ptr; }

    /** \brief Empty grid constructor. */
    Grid(): buffer(nullptr), dims{0}, offs{0} {

    }

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

    /** \brief Grid constructor
     *
     * Provide pointer and dimensions array specified as arguments. sizes must contain NumDims values.
    */
    Grid(Dtype *const d, size_t *sizes):
      buffer(d) {
      offs[NumDims-1] = 1;
      #pragma unroll
      for(int i = NumDims-1; i > 0; i--) {
        dims[i] = sizes[i];
        offs[i-1] = dims[i]*offs[i];
      }
      dims[0] = sizes[0];
    }

    Grid(const Grid&) = default;
    ~Grid() = default;

    /** \brief Bracket indexing.
     *
     *  Accessing data this way will be safe (indices are checked) and convenient,
     *  but not maximally efficient (unless the compiler is really good).
     *  Use operator() for fastest (but unchecked) access or access data directly.
     */
    CUDA_CALLABLE_MEMBER subgrid_t operator[](size_t i) const {
      check_index(i, dims[0]);
      return Grid<Dtype,NumDims-1,isCUDA>(*this, i);
    }

    /** \brief Initializer list indexing
     *
     */
    template<typename... I>
    CUDA_CALLABLE_MEMBER inline Dtype& operator()(I... indices) {
      return buffer[getPos(indices...)];
    }

    template<typename... I>
    CUDA_CALLABLE_MEMBER inline Dtype operator()(I... indices) const {
      return buffer[getPos(indices...)];
    }

    /** \brief Return memory address of specified index */
    template<typename... I>
    CUDA_CALLABLE_MEMBER inline Dtype* address(I... indices) {
      return &buffer[getPos(indices...)];
    }

    template<typename... I>
    CUDA_CALLABLE_MEMBER inline const Dtype* address(I... indices) const {
      return &buffer[getPos(indices...)];
    }

    /** \brief copy contents to dest
     *
     *  Sizes should be the same, but will narrow as necessary.  Will copy across device/host.
     */
    template<bool destCUDA>
    size_t copyTo(Grid<Dtype,NumDims,destCUDA>& dest) const {
      size_t sz = std::min(size(), dest.size());
      if(sz == 0) return 0;
      cudaMemcpyKind kind = copyKind(isCUDA,destCUDA);
      LMG_CUDA_CHECK(cudaMemcpy(dest.data(),data(),sz*sizeof(Dtype),kind));
      return sz;
    }

    /** \brief copy contents from src
     *
     *  Sizes should be the same, but will narrow as necessary.  Will copy across device/host.
     */
    template<bool srcCUDA>
    size_t copyFrom(const Grid<Dtype,NumDims,srcCUDA>& src) {
      size_t sz = std::min(size(), src.size());
      if(sz == 0) return 0;
      cudaMemcpyKind kind = copyKind(srcCUDA,isCUDA);
      LMG_CUDA_CHECK(cudaMemcpy(data(),src.data(),sz*sizeof(Dtype),kind));
      return sz;
    }

    /** \brief Set contents to zero.
     *
     */
    void fill_zero() {
      if(isCUDA) cudaMemset(data(), 0, sizeof(Dtype)*size());
      else memset(data(), 0, sizeof(Dtype)*size());
    }

    // constructor used by operator[], create a subgrid, assuming memory is allocated
    CUDA_CALLABLE_MEMBER
    explicit Grid(const Grid<Dtype,NumDims+1,isCUDA>& G, size_t i): buffer(G.data() ? &G.data()[i*G.offset(0)] : nullptr) {
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

    CUDA_CALLABLE_MEMBER void check_index(size_t i, size_t dim) const {
#ifndef __CUDA_ARCH__
      if(i >= dim) {
        throw std::out_of_range("Invalid range. "+ itoa(i) + " >= " + itoa(dim));
      }
#else
      assert(i < dim);
#endif
    }
  public:
    using type = Dtype;
    using subgrid_t = Dtype;
    using managed_t = ManagedGrid<Dtype,1>;
    using cpu_grid_t = Grid<Dtype,1,false>;
    using gpu_grid_t = Grid<Dtype,1,true>;

    static constexpr size_t N = 1;
    static constexpr bool GPU = isCUDA;

    /// dimensions along each axis
    CUDA_CALLABLE_MEMBER inline const size_t * dimensions() const { return dims; }
    /// dimensions along specified axis
    CUDA_CALLABLE_MEMBER inline size_t dimension(size_t i) const { check_index(i, 1); return dims[i]; }

    /// number of elements in grid
    CUDA_CALLABLE_MEMBER inline size_t size() const { return dims[0]; }

    /// pointer to underlying data
    CUDA_CALLABLE_MEMBER inline Dtype * data() const { return buffer; }

    CUDA_CALLABLE_MEMBER inline size_t offset(size_t i) const { return 1; }

    /// set the underlying memory buffer - use with caution!
    CUDA_CALLABLE_MEMBER inline void set_buffer(Dtype *ptr) { buffer = ptr; }

    Grid(): buffer(nullptr), dims{0} { }

    Grid(Dtype* const d, size_t sz):
      buffer(d), dims{sz} { }

    CUDA_CALLABLE_MEMBER inline Dtype& operator[](size_t i) {
      check_index(i,dims[0]);
      return buffer[i];
    }

    CUDA_CALLABLE_MEMBER inline Dtype operator[](size_t i) const {
      check_index(i,dims[0]);
      return buffer[i];
    }

    CUDA_CALLABLE_MEMBER inline Dtype& operator()(size_t a) {
      return buffer[a];
    }

    CUDA_CALLABLE_MEMBER inline Dtype operator()(size_t a) const {
      return buffer[a];
    }

    void fill_zero() {
      if(isCUDA) cudaMemset(data(), 0, sizeof(Dtype)*size());
      else memset(data(), 0, sizeof(Dtype)*size());
    }

    //only called from regular Grid
    CUDA_CALLABLE_MEMBER
    explicit Grid<Dtype,1,isCUDA>(const Grid<Dtype,2,isCUDA>& G, size_t i):
      buffer(&G.data()[i*G.offset(0)]), dims{G.dimension(1)} {}


    template<bool destCUDA>
    size_t copyTo(Grid<Dtype,1,destCUDA>& dest) const {
      size_t sz = std::min(size(), dest.size());
      if(sz == 0) return sz;
      cudaMemcpyKind kind = copyKind(isCUDA,destCUDA);
      LMG_CUDA_CHECK(cudaMemcpy(dest.data(),data(),sz*sizeof(Dtype),kind));
      return sz;
    }

    template<bool srcCUDA>
    size_t copyFrom(const Grid<Dtype,1,srcCUDA>& src) {
      size_t sz = std::min(size(), src.size());
      if(sz == 0) return sz;
      cudaMemcpyKind kind = copyKind(srcCUDA,isCUDA);
      LMG_CUDA_CHECK(cudaMemcpy(data(),src.data(),sz*sizeof(Dtype),kind));
      return sz;
    }

};

#define EXPAND_GRID_DEFINITIONS(Z,SIZE,_) \
typedef Grid<float, SIZE, false> Grid##SIZE##f; \
typedef Grid<double, SIZE, false> Grid##SIZE##d; \
typedef Grid<float, SIZE, true> Grid##SIZE##fCUDA; \
typedef Grid<double, SIZE, true> Grid##SIZE##dCUDA;

BOOST_PP_REPEAT_FROM_TO(1,LIBMOLGRID_MAX_GRID_DIM, EXPAND_GRID_DEFINITIONS, 0);

}

#endif /* GRID_H_ */
