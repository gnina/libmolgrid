/** \file managed_grid.h
 *
 * A grid that manages its own memory using a shared pointer.
 * Any libmolgrid routines that create a grid option (e.g. readers)
 * return a ManagedGrid.  Memory is first allocated as CPU memory
 * but can be explicitly converted to GPU memory and back.
 *
 */

#ifndef MANAGED_GRID_H_
#define MANAGED_GRID_H_

#include <memory>
#include <utility>
#include <boost/lexical_cast.hpp>

#include "libmolgrid/grid.h"


namespace libmolgrid {


template<typename Dtype>
struct mgrid_buffer_data {
    Dtype *gpu_ptr;
    bool sent_to_gpu;
};

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
    mutable gpu_grid_t gpu_grid; //treated as a cache
    cpu_grid_t cpu_grid;
    std::shared_ptr<Dtype> cpu_ptr; //shared pointer lets us not worry about copying the grid
    size_t capacity = 0; //amount of memory allocated (for resizing)

    using buffer_data = mgrid_buffer_data<Dtype>;
    mutable buffer_data *gpu_info = nullptr;


    ///empty (unusable) grid
    ManagedGridBase() = default;

    // deallocate our special buffer memory, include gpu memory if present
    static void delete_buffer(void *ptr) {
      buffer_data *data = (buffer_data*)(ptr) - 1;
      if(data->gpu_ptr != nullptr) {
        //deallocate gpu
        cudaFree(data->gpu_ptr);
      }
      free(data);
    }
    //allocate and set the cpu pointer (and grid) with space for sent_to_gpu bool, set the bool ptr location
    //does not initialize memory
    void alloc_and_set_cpu(size_t sz) {
      //put buffer data at start so know where it is on delete
      void *buffer = malloc(sizeof(buffer_data)+sz*sizeof(Dtype));
      Dtype *cpu_data = (Dtype*)((buffer_data*)buffer+1);

      if(!buffer) throw std::runtime_error("Could not allocate "+itoa(sz*sizeof(Dtype))+" bytes of CPU memory in ManagedGrid");
      cpu_ptr = std::shared_ptr<Dtype>(cpu_data, delete_buffer);
      cpu_grid.set_buffer(cpu_ptr.get());
      gpu_info = (buffer_data*)buffer;
      gpu_info->gpu_ptr = nullptr;
      gpu_info->sent_to_gpu = false;
    }

    //allocate and set gpu_ptr and grid, does not initialize memory, should not be called if memory is already allocated
    void alloc_and_set_gpu(size_t sz) const {
      if(gpu_info == nullptr)
        throw std::runtime_error("Attempt to allocate gpu memory in empty ManagedGrid");
      if(gpu_info->gpu_ptr != nullptr) {
        throw std::runtime_error("Attempt to reallocate gpu memory in  ManagedGrid");
      }
      //we are not actually using unified memory, but this make debugging easier?
      cudaError_t err = cudaMalloc(&gpu_info->gpu_ptr,sz*sizeof(Dtype));
      cudaGetLastError();
      if(err != cudaSuccess) {
        throw std::runtime_error("Could not allocate "+itoa(sz*sizeof(Dtype))+" bytes of GPU memory in ManagedGrid");
      }
      gpu_grid.set_buffer(gpu_info->gpu_ptr);
    }

    template<typename... I, typename = typename std::enable_if<sizeof...(I) == NumDims>::type>
    ManagedGridBase(I... sizes): gpu_grid(nullptr, sizes...), cpu_grid(nullptr, sizes...) {
      //allocate buffer
      capacity = this->size();
      alloc_and_set_cpu(capacity); //even with capacity == 0 need to allocate sent_to_gpu
      memset(cpu_ptr.get(), 0, capacity*sizeof(Dtype));
      gpu_info->sent_to_gpu = false;
    }

    //helper for clone, allocate new memory and copies contents of current ptr into it
    void clone_ptrs() {
      if(capacity == 0) {
        return;
      }

      //duplicate cpu memory and set sent_to_gpu
      std::shared_ptr<Dtype> old = cpu_ptr;
      buffer_data oldgpu = *gpu_info;
      alloc_and_set_cpu(capacity);
      memcpy(cpu_ptr.get(), old.get(), sizeof(Dtype)*capacity);
      gpu_info->sent_to_gpu = oldgpu.sent_to_gpu;

      //if allocated, duplicate gpu, but only if it is active
      if(oldgpu.gpu_ptr && oldgpu.sent_to_gpu) {
        alloc_and_set_gpu(capacity);
        LMG_CUDA_CHECK(cudaMemcpy(gpu_info->gpu_ptr, oldgpu.gpu_ptr, sizeof(Dtype)*capacity, cudaMemcpyDeviceToDevice));
      }
    }
  public:

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

    /// set contents to zero
    inline void fill_zero() {
      if(ongpu()) gpu_grid.fill_zero();
      else cpu_grid.fill_zero();
    }

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

    /** \brief Copy data into dest.  Should be same size, but will narrow if needed */
    size_t copyTo(cpu_grid_t& dest) const {
      size_t sz = std::min(size(), dest.size());
      if(sz == 0) return 0;
      if(ongpu()) {
        LMG_CUDA_CHECK(cudaMemcpy(dest.data(), gpu_grid.data(), sz*sizeof(Dtype), cudaMemcpyDeviceToHost));
      } else { //host ot host
        memcpy(dest.data(),cpu_grid.data(),sz*sizeof(Dtype));
      }
      return sz;
    }

    /** \brief Copy data into dest.  Should be same size, but will narrow if needed */
    size_t copyTo(gpu_grid_t& dest) const {
      size_t sz = std::min(size(), dest.size());
      if(sz == 0) return 0;
      if(ongpu()) {
        LMG_CUDA_CHECK(cudaMemcpy(dest.data(),gpu_grid.data(),sz*sizeof(Dtype),cudaMemcpyDeviceToDevice));
      } else {
        LMG_CUDA_CHECK(cudaMemcpy(dest.data(),cpu_grid.data(),sz*sizeof(Dtype),cudaMemcpyHostToDevice));
      }
      return sz;
    }

    /** \brief Copy data into dest.  Should be same size, but will narrow if needed */
    size_t copyTo(ManagedGridBase<Dtype, NumDims>& dest) const {
      if(dest.ongpu()) {
        return copyTo(dest.gpu_grid);
      } else {
        return copyTo(dest.cpu_grid);
      }
    }

    /** \brief Copy data from src.  Should be same size, but will narrow if needed */
    size_t copyFrom(const cpu_grid_t& src) {
      size_t sz = std::min(size(), src.size());
      if(sz == 0) return 0;
      if(ongpu()) {
       LMG_CUDA_CHECK(cudaMemcpy(gpu_grid.data(), src.data(), sz*sizeof(Dtype), cudaMemcpyHostToDevice));
      } else {
        memcpy(cpu_grid.data(),src.data(),sz*sizeof(Dtype));
      }
      return sz;
    }

    /** \brief Copy data from src.  Should be same size, but will narrow if needed */
    size_t copyFrom(const gpu_grid_t& src) {
      size_t sz = std::min(size(), src.size());
      if(sz == 0) return 0;
      if(ongpu()) {
        LMG_CUDA_CHECK(cudaMemcpy(gpu_grid.data(),src.data(),sz*sizeof(Dtype),cudaMemcpyDeviceToDevice));
      } else {
        LMG_CUDA_CHECK(cudaMemcpy(cpu_grid.data(),src.data(),sz*sizeof(Dtype),cudaMemcpyDeviceToHost));
      }
      return sz;
    }

    /** \brief Copy data from src.  Should be same size, but will narrow if needed */
    size_t copyFrom(const ManagedGridBase<Dtype, NumDims>& src) {
      if(src.ongpu()) {
        return copyFrom(src.gpu_grid);
      } else { //on host
        return copyFrom(src.cpu_grid);
      }
    }

    /** \brief Copy data from src into this starting at start.  Should be same size, but will narrow if needed */
    size_t copyInto(size_t start, const ManagedGridBase<Dtype, NumDims>& src) {
      size_t off = offset(0)*start;
      size_t sz = size()-off;
      sz = std::min(sz, src.size());
      if(sz == 0) return 0;
      if(src.ongpu()) {
        if(ongpu()) {
          LMG_CUDA_CHECK(cudaMemcpy(gpu_grid.data()+off,src.gpu_grid.data(),sz*sizeof(Dtype),cudaMemcpyDeviceToDevice));
        } else {
          LMG_CUDA_CHECK(cudaMemcpy(cpu_grid.data()+off,src.gpu_grid.data(),sz*sizeof(Dtype),cudaMemcpyDeviceToHost));
        }
      } else { //on host
        if(ongpu()) {
         LMG_CUDA_CHECK(cudaMemcpy(gpu_grid.data()+off, src.data(), sz*sizeof(Dtype), cudaMemcpyHostToDevice));
        } else {
          memcpy(cpu_grid.data()+off,src.data(),sz*sizeof(Dtype));
        }
      }
      return sz;
    }

    /** \brief Return a grid in the specified shape that attempts to reuse the memory of this grid.
     * Memory will be allocated if needed.  Data will be truncated/copied as needed.
     * DANGER!  The returned grid may or may not mirror this grid depending on the shape.
     * This function is provided so code can be optimized to avoid unnecessary allocations and
     * should be used carefully.
     */
    template<typename... I, typename = typename std::enable_if<sizeof...(I) == NumDims>::type>
    ManagedGrid<Dtype, NumDims> resized(I... sizes) {
      cpu_grid_t g(nullptr, sizes...);
      if(g.size() <= capacity) {
        //no need to allocate and copy; capacity stays the same
        ManagedGrid<Dtype, NumDims> tmp;
        tmp.cpu_ptr = cpu_ptr;
        tmp.gpu_info = gpu_info;
        tmp.cpu_grid = cpu_grid_t(cpu_ptr.get(), sizes...);
        if(gpu_info) tmp.gpu_grid = gpu_grid_t(gpu_info->gpu_ptr, sizes...);
        tmp.capacity = capacity;
        return tmp;
      } else {
        ManagedGrid<Dtype, NumDims> tmp(sizes...);
        if(size() > 0 && tmp.size() > 0) {
          if(ongpu()) tmp.togpu(); //allocate gpu memory
          copyTo(tmp);
        }
        return tmp;
      }
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

    /** \brief Transfer data to GPU */
    void togpu(bool dotransfer=true) const {
      if(capacity == 0) return;
      //check that memory is allocated - even if data is on gpu, may still need to set this mgrid's gpu_grid
      if(gpu_grid.data() == nullptr) {
        if(gpu_info->gpu_ptr == nullptr) {
          alloc_and_set_gpu(capacity);
        } //otherwise some other copy has already allocated memory, just need to set
        size_t offset = cpu_grid.data() - cpu_ptr.get(); //might be subgrid
          gpu_grid.set_buffer(gpu_info->gpu_ptr+offset);
      }
      if(oncpu() && dotransfer) {
        LMG_CUDA_CHECK(cudaMemcpy(gpu_info->gpu_ptr,cpu_ptr.get(),capacity*sizeof(Dtype),cudaMemcpyHostToDevice));
      }
      if(gpu_info) gpu_info->sent_to_gpu = true;
    }

    /** \brief Transfer data to CPU.  If not dotransfer, data is not copied back. */
    void tocpu(bool dotransfer=true) const {
      if(ongpu() && capacity > 0 && dotransfer) {
        LMG_CUDA_CHECK(cudaMemcpy(cpu_ptr.get(),gpu_info->gpu_ptr,capacity*sizeof(Dtype),cudaMemcpyDeviceToHost));
      }
      if(gpu_info) gpu_info->sent_to_gpu = false;
    }

    /** \brief Return true if memory is currently on GPU */
    bool ongpu() const {
      bool ret = gpu_info && gpu_info->sent_to_gpu;
      if(ret && gpu_grid.data() == nullptr) togpu(); //needed if this is a copy made before another transfered
      return ret;
    }

    /** \brief Return true if memory is currently on CPU */
    bool oncpu() const { return gpu_info == nullptr || !gpu_info->sent_to_gpu; }


    operator cpu_grid_t() const { return cpu(); }
    operator cpu_grid_t&() {return cpu(); }

    operator gpu_grid_t() const { return gpu(); }
    operator gpu_grid_t&() {return gpu(); }

    /// Return pointer to CPU data
    inline const Dtype * data() const { tocpu(); return cpu().data(); }
    inline Dtype * data() { tocpu(); return cpu().data(); }

    //pointer equality
    bool operator==(const ManagedGridBase<Dtype, NumDims>& rhs) const {
      return cpu_ptr == rhs.cpu_ptr;
    }
  protected:
    // constructor used by operator[]
    friend ManagedGridBase<Dtype,NumDims-1>;
    explicit ManagedGridBase(const ManagedGridBase<Dtype,NumDims+1>& G, size_t i):
      gpu_grid(G.gpu_grid, i), cpu_grid(G.cpu_grid, i), cpu_ptr(G.cpu_ptr),
      capacity(G.capacity), gpu_info(G.gpu_info) {}
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
    using base_t = ManagedGridBase<Dtype, NumDims>;

    //empty, unusable grid
    ManagedGrid() = default;

    template<typename... I, typename = typename std::enable_if<sizeof...(I) == NumDims>::type>
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

    /** \brief Return a copy of this grid */
    ManagedGrid<Dtype, NumDims> clone() const {
      ManagedGrid<Dtype, NumDims> ret(*this);
      ret.clone_ptrs();
      return ret;
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
    using base_t = ManagedGridBase<Dtype, 1>;

    ManagedGrid() = default;
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

    ManagedGrid<Dtype, 1> clone() const {
      ManagedGrid<Dtype, 1> ret(*this);
      ret.clone_ptrs();
      return ret;
    }

  protected:
    //only called from regular Grid
    friend ManagedGrid<Dtype,2>;
    explicit ManagedGrid<Dtype,1>(const ManagedGridBase<Dtype,2>& G, size_t i):
      ManagedGridBase<Dtype,1>(G, i) {}

};


#define EXPAND_MGRID_DEFINITIONS(Z,SIZE,_) \
typedef ManagedGrid<float, SIZE> MGrid##SIZE##f; \
typedef ManagedGrid<double, SIZE> MGrid##SIZE##d;


BOOST_PP_REPEAT_FROM_TO(1,LIBMOLGRID_MAX_GRID_DIM, EXPAND_MGRID_DEFINITIONS, 0);

}
#endif /* MANAGED_GRID_H_ */
