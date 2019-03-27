/** \file managed_grid.h
 *
 * A grid that manages its own memory using a shared pointer.
 * Any libmolgrid routines that create a grid option (e.g. readers)
 * return a ManagedGrid.  Memory is allocated as unified CUDA memory.
 *
 */

#ifndef MANAGED_GRID_H_
#define MANAGED_GRID_H_

#include <memory>
#include <grid.h>
#include <boost/lexical_cast.hpp>


namespace libmolgrid {



/** \brief ManagedGrid base class */
template<typename Dtype, std::size_t NumDims>
class ManagedGridBase {
  public:
    using gpu_grid_t = Grid<Dtype, NumDims, true>; ///cuda grid type
    using cpu_grid_t = Grid<Dtype, NumDims, false>;
    using type = Dtype;
    static constexpr size_t N = NumDims;

  protected:
    //two different views of the same memory
    gpu_grid_t gpu_grid;
    cpu_grid_t cpu_grid;
    std::shared_ptr<Dtype> ptr; //shared pointer lets us not worry about copying the grid
    mutable bool sent_to_gpu = false; //a CUDA grid view has been requested and there have been no host accesses

    template<typename... I>
    ManagedGridBase(I... sizes): gpu_grid(nullptr, sizes...), cpu_grid(nullptr, sizes...) {
      //allocate buffer
      ptr = create_unified_shared_ptr<Dtype>(this->size());
      gpu_grid.set_buffer(ptr.get());
      cpu_grid.set_buffer(ptr.get());
    }

  public:
    const std::shared_ptr<Dtype> pointer() const { return ptr; }

    /// dimensions along each axis
    inline const size_t * dimensions() const { return cpu_grid.dimensions(); }
    /// dimensions along specified axis
    inline size_t dimension(size_t i) const { return cpu_grid.dimension(i); }

    /// offset for each dimension, all indexing calculations use this
    inline const size_t * offsets() const { return cpu_grid.offsets(); }
    /// offset for each dimension, all indexing calculations use this
    inline size_t offset(size_t i) const { return cpu_grid.offset(i); }

    /// number of elements in grid
    inline size_t size() const { return cpu_grid.size(); }

    /// raw pointer to underlying data - for subgrids may point into middle of grid
    CUDA_CALLABLE_MEMBER inline Dtype * data() const { return cpu_grid.data(); }

    /** \brief Initializer list indexing
     *
     */
    template<typename... I>
    inline Dtype& operator()(I... indices) {
      tocpu();
      return cpu_grid(indices...);
    }

    template<typename... I>
    inline Dtype operator()(I... indices) const {
      tocpu();
      return cpu_grid(indices...);
    }

    /** \brief Copy data into dest.  Should be same size, but will narrow if neede */
    void copyTo(cpu_grid_t& dest) const {
      tocpu();
      size_t sz = std::min(size(), dest.size());
      memcpy(dest.data(),cpu_grid.data(),sz*sizeof(Dtype));
    }

    /** \brief Copy data into dest.  Should be same size, but will narrow if neede */
    void copyTo(gpu_grid_t& dest) const {
      togpu();
      size_t sz = std::min(size(), dest.size());
      cudaMemcpy(dest.data(),gpu_grid.data(),sz*sizeof(Dtype),cudaMemcpyDeviceToDevice);
    }

    /** \brief Copy data from src.  Should be same size, but will narrow if needed */
    void copyFrom(const cpu_grid_t& dest) {
      tocpu();
      size_t sz = std::min(size(), dest.size());
      memcpy(cpu_grid.data(),dest.data(),sz*sizeof(Dtype));
    }

    /** \brief Copy data from src.  Should be same size, but will narrow if neede */
    void copyFrom(const gpu_grid_t& dest) {
      togpu();
      size_t sz = std::min(size(), dest.size());
      cudaMemcpy(gpu_grid.data(),dest.data(),sz*sizeof(Dtype),cudaMemcpyDeviceToDevice);
    }

    /** \brief Return GPU Grid view.  Host code should not access the grid
     * until the GPU code is complete.
     */
    const gpu_grid_t& gpu() const { togpu(); return gpu_grid; }
    gpu_grid_t& gpu() { togpu(); return gpu_grid; }

    /** \brief Return CPU Grid view.  GPU code should no longer access this memory.
     */
    const cpu_grid_t& cpu() const { tocpu(); return cpu_grid; }
    cpu_grid_t& cpu() { tocpu(); return cpu_grid; }

    /** \brief Indicate memory is being worked on by GPU */
    void togpu() const {
      if(!sent_to_gpu) sync(); //only sync if changing state
      sent_to_gpu = true;
    }

    /** \brief Indicate memory is being worked on by CPU */
    void tocpu() const {
      if(sent_to_gpu) sync();
      sent_to_gpu = false;
    }

    /** \brief Return true if memory is currently on GPU */
    bool ongpu() const { return sent_to_gpu; }

    /** \brief Return true if memory is currently on CPU */
    bool oncpu() const { return !sent_to_gpu; }

    /** \brief Synchronize gpu memory
     *  If operated on as device memory, must call synchronize before accessing on host.
     */
    void sync() const {
      cudaDeviceSynchronize();
    }

    operator cpu_grid_t() const { return cpu(); }
    operator cpu_grid_t&() {return cpu(); }

    operator gpu_grid_t() const { return gpu(); }
    operator gpu_grid_t&() {return gpu(); }

    //pointer equality
    bool operator==(const ManagedGridBase<Dtype, NumDims>& rhs) const {
      return ptr == rhs.ptr;
    }
  protected:
    // constructor used by operator[]
    friend ManagedGridBase<Dtype,NumDims-1>;
    explicit ManagedGridBase(const ManagedGridBase<Dtype,NumDims+1>& G, size_t i):
      gpu_grid(G.gpu_grid, i), cpu_grid(G.cpu_grid, i), ptr(G.ptr), sent_to_gpu(G.sent_to_gpu) {}


};

/** \brief A dense grid whose memory is managed by the class.
 *
 * Memory is allocated as unified memory so it can be safely accessed
 * from either the CPU or GPU.  Note that while the memory is accessible
 * on the GPU, ManagedGrid objects should only be used directly on the host.
 * Device code should use a Grid view of the ManagedGrid.  Accessing ManagedGrid
 * data from the host while the GPU is writing to the grid will result in
 * undefined behavior.
 *
 * If CUDA fails to allocate unified memory (presumably due to lack of GPUs),
 * host-only memory will be used instead.
 *
 * Grid is a container class of ManagedGrid instead of a base class.  Explicit
 * transformation to a Grid view is required to clearly indicate how the memory
 * should be accessed.  This can be done with cpu and gpu methods or by an
 * explicit cast to Grid.
 *
 * There are two class specialization to support bracket indexing.
 */
template<typename Dtype, std::size_t NumDims>
class ManagedGrid :  public ManagedGridBase<Dtype, NumDims> {
  public:
    using subgrid_t = ManagedGrid<Dtype,NumDims-1>;

    template<typename... I>
    ManagedGrid(I... sizes): ManagedGridBase<Dtype,NumDims>(sizes...) {
    }

    /** \brief Bracket indexing.
     *
     *  Accessing data this way will be safe (indices are checked) and convenient,
     *  but not maximally efficient (unless the compiler is really good).
     *  Use operator() for fastest (but unchecked) access or access data directly.
     */
    subgrid_t operator[](size_t i) const {
      if(i >= this->cpu_grid.dimension(0))
        throw std::out_of_range("Index "+boost::lexical_cast<std::string>(i)+" out of bounds of dimension "+boost::lexical_cast<std::string>(this->cpu_grid.dimension(0)));
      return ManagedGrid<Dtype,NumDims-1>(*static_cast<const ManagedGridBase<Dtype, NumDims> *>(this), i);
    }

  protected:
    //only called from regular Grid
    friend ManagedGrid<Dtype,NumDims+1>;
    explicit ManagedGrid<Dtype,NumDims>(const ManagedGridBase<Dtype,NumDims+1>& G, size_t i):
      ManagedGridBase<Dtype,NumDims>(G, i) {}

};


// class specialization of managed grid to make final operator[] return scalar
template<typename Dtype >
class ManagedGrid<Dtype, 1> : public ManagedGridBase<Dtype, 1> {
  public:
    using subgrid_t = Dtype;

    ManagedGrid(size_t sz): ManagedGridBase<Dtype, 1>(sz) {
    }

    inline Dtype& operator[](size_t i) {
      this->tocpu();
      return this->cpu_grid[i];
    }

    inline Dtype operator[](size_t i) const {
      this->tocpu();
      return this->cpu_grid[i];
    }

    inline Dtype& operator()(size_t a) {
      this->tocpu();
      return this->cpu_grid(a);
    }

    inline Dtype operator()(size_t a) const {
      this->tocpu();
      return this->cpu_grid(a);
    }

  protected:
    //only called from regular Grid
    friend ManagedGrid<Dtype,2>;
    explicit ManagedGrid<Dtype,1>(const ManagedGridBase<Dtype,2>& G, size_t i):
      ManagedGridBase<Dtype,1>(G, i) {}

};

#if 0
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
  //not all GPUs support prefetch, so absorb any errors here
  cudaGetLastError();
  //make sure mem is synced
  if(isCUDA) mg.gpu();
  else mg.cpu();
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
  //not all GPUs support prefetch, so absorb any errors here
  cudaGetLastError();
  //make sure mem is synced
  if(isCUDA) mg.gpu();
  else mg.cpu();
}
#endif

#define EXPAND_MGRID_DEFINITIONS(Z,SIZE,_) \
typedef ManagedGrid<float, SIZE> MGrid##SIZE##f; \
typedef ManagedGrid<double, SIZE> MGrid##SIZE##d;


BOOST_PP_REPEAT_FROM_TO(1,LIBMOLGRID_MAX_GRID_DIM, EXPAND_MGRID_DEFINITIONS, 0);

}
#endif /* MANAGED_GRID_H_ */
