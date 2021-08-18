#include "libmolgrid/grid_maker.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

namespace libmolgrid {
    __shared__ uint scanScratch[LMG_CUDA_NUM_THREADS * 2];
    __shared__ uint scanBuffer[2][LMG_CUDA_NUM_THREADS];
    __shared__ uint scanOutput[LMG_CUDA_NUM_THREADS];
    __shared__ uint atomIndices[LMG_CUDA_NUM_THREADS];
    __shared__ uint atomMask[LMG_CUDA_NUM_THREADS];

    //TODO: warp shuffle version
    inline __device__ uint warpScanInclusive(int threadIndex, uint idata,
        volatile uint *s_Data, uint size) {
      uint pos = 2 * threadIndex - (threadIndex & (size - 1));
      s_Data[pos] = 0;
      pos += size;
      s_Data[pos] = idata;
    
      for (uint offset = 1; offset < size; offset <<= 1)
        s_Data[pos] += s_Data[pos - offset];
    
      return s_Data[pos];
    }
    
    inline __device__ uint warpScanExclusive(int threadIndex, uint idata,
        volatile uint *sScratch, uint size) {
      return warpScanInclusive(threadIndex, idata, sScratch, size) - idata;
    }
    
    __inline__ __device__ void sharedMemExclusiveScan(int threadIndex, uint* sInput,
        uint* sOutput) {
      uint idata = sInput[threadIndex];
      //Bottom-level inclusive warp scan
      uint warpResult = warpScanInclusive(threadIndex, idata, scanScratch,
          WARP_SIZE);

      // Save top elements of each warp for exclusive warp scan sync
      // to wait for warp scans to complete (because s_Data is being
      // overwritten)
      __syncthreads();
    
      if ((threadIndex & (WARP_SIZE - 1)) == (WARP_SIZE - 1)) {
        scanScratch[threadIndex >> LOG2_WARP_SIZE] = warpResult;
      }
    
      // wait for warp scans to complete
      __syncthreads();
    
      if (threadIndex < (LMG_CUDA_NUM_THREADS / WARP_SIZE)) {
        // grab top warp elements
        uint val = scanScratch[threadIndex];
        // calculate exclusive scan and write back to shared memory
        scanScratch[threadIndex] = warpScanExclusive(threadIndex, val, scanScratch,
            LMG_CUDA_NUM_THREADS >> LOG2_WARP_SIZE);
      }
    
      //return updated warp scans with exclusive scan results
      __syncthreads();
    
      sOutput[threadIndex] = warpResult + scanScratch[threadIndex >> LOG2_WARP_SIZE]
          - idata;
    }
    

    uint2 GridMaker::get_bounds_1d(const float grid_origin,
        float coord, float densityrad) const {
      uint2 bounds{0, 0};
      float low = coord - densityrad - grid_origin;
      if (low > 0) {
        bounds.x = floor(low / resolution);
      }

      float high = coord + densityrad - grid_origin;
      if (high > 0) { //otherwise zero
        bounds.y = min(dim, (unsigned) ceil(high / resolution));
      }
      return bounds;
    }

    //return squared distance between pt and (x,y,z)
    __host__ __device__ inline
    float sqDistance(float3 pt, float x, float y, float z) {
      float ret;
      float tmp = pt.x - x;
      ret = tmp * tmp;
      tmp = pt.y - y;
      ret += tmp * tmp;
      tmp = pt.z - z;
      ret += tmp * tmp;
      return ret;
    }

    //non-binary, gaussian case
    template<>
    float GridMaker::calc_point<false>(float ax, float ay, float az, float ar,
        const float3& grid_coords) const {
      float rsq = sqDistance(grid_coords, ax, ay, az);
      ar *= radius_scale;
      //For non-binary density we want a Gaussian where 2 std occurs at the
      //radius, after which it becomes quadratic.
      //The quadratic is fit to have both the same value and first derivative
      //at the cross over point and a value and derivative of zero at fianl_radius_multiple
      float dist = sqrtf(rsq);
      if (dist > ar * final_radius_multiple) {
        return 0.0;
      } else
      if (dist <= ar * gaussian_radius_multiple) {
        //return gaussian
        float ex = -2.0 * dist * dist / (ar*ar);
        return exp(ex);
      } else { //return quadratic
        float dr = dist / ar;
        float q = (A * dr + B) * dr + C;
        return q > 0 ? q : 0; //avoid very small negative numbers
      }

    }

    template<>
    float GridMaker::calc_point<true>(float ax, float ay, float az, float ar,
        const float3& grid_coords) const {
      float rsq = sqDistance(grid_coords, ax, ay, az);
      ar *= radius_scale;
      //is point within radius?
      if (rsq < ar * ar)
        return 1.0;
      else
        return 0.0;
    }

    /* \brief The GPU forward code path launches a kernel (forward_gpu) that
     * sets the grid values in two steps: first each thread cooperates with the
     * other threads in its block to determine which atoms could possibly
     * overlap them. Then, working from this significantly reduced atom set,
     * they actually check whether they are overlapped by an atom and update
     * their density accordingly. atomOverlapsBlock is a helper for generating
     * the reduced array of possibly relevant atoms.
     * @param[in] atom index to check
     * @param[in] grid origin
     * @param[in] coordinates (Nx3)
     * @param[in] type indices (N integers stored as floats)
     * @param[in] radii (N)
     * @param[out] 1 if atom could overlap block, 0 if not
     */
    __device__
    static unsigned atom_overlaps_block(unsigned aidx, float3& grid_origin,
        float resolution, const float3 *coords, float radius, float rmult) {

      unsigned xi = blockIdx.x * blockDim.x;
      unsigned yi = blockIdx.y * blockDim.y;
      unsigned zi = blockIdx.z * blockDim.z;
    
      //compute corners of block
      float startx = xi * resolution + grid_origin.x;
      float starty = yi * resolution + grid_origin.y;
      float startz = zi * resolution + grid_origin.z;
    
      float endx = startx + resolution * blockDim.x;
      float endy = starty + resolution * blockDim.y;
      float endz = startz + resolution * blockDim.z;

      float3 a = coords[aidx];
      float centerx = a.x;
      float centery = a.y;
      float centerz = a.z;
      float r = radius * rmult;
    
      //does atom overlap box?
      return !((centerx - r > endx) || (centerx + r < startx)
          || (centery - r > endy) || (centery + r < starty)
          || (centerz - r > endz) || (centerz + r < startz));
    }


    template <typename Dtype, bool Binary>
    __device__ void GridMaker::set_atoms(unsigned rel_atoms, float3 grid_origin,
        const float3 *coord_data, const float *tdata, const float *radii, Dtype *data) {
      //figure out what grid point we are 
      unsigned xi = threadIdx.x + blockIdx.x * blockDim.x;
      unsigned yi = threadIdx.y + blockIdx.y * blockDim.y;
      unsigned zi = threadIdx.z + blockIdx.z * blockDim.z;

      if(xi >= dim || yi >= dim || zi >= dim)
        return;//bail if we're off-grid, this should not be common

      //compute x,y,z coordinate of grid point
      float3 grid_coords;
      grid_coords.x = xi * resolution + grid_origin.x;
      grid_coords.y = yi * resolution + grid_origin.y;
      grid_coords.z = zi * resolution + grid_origin.z;
      unsigned goffset = ((xi*dim)+yi)*dim + zi; //offset into channel grid
      unsigned chmult = dim*dim*dim; //what to multiply type/channel seletion by

      //iterate over all possibly relevant atoms
      for(unsigned ai = 0; ai < rel_atoms; ai++) {
        unsigned i = atomIndices[ai];
        float3 c = coord_data[i];
        float val = calc_point<Binary>(c.x, c.y, c.z, radii[i], grid_coords);
        int atype = int(tdata[i]); //type is assumed correct because atom_overlaps at least gets rid of neg

        if(Binary) {
            if(val != 0)
              data[atype*chmult+goffset] = 1.0;
        } else if(val > 0) {
          data[atype*chmult+goffset] += val;
        }

      }
    }

    template <typename Dtype, bool Binary>
    __global__ void
  //  __launch_bounds__(LMG_CUDA_NUM_THREADS)
    forward_gpu(GridMaker gmaker, float3 grid_origin,
        const Grid<float, 2, true> coords, const Grid<float, 1, true> type_index,
        const Grid<float, 1, true> radii, Grid<Dtype, 4, true> out) {
      //this is the thread's index within its block, used to parallelize over atoms
      unsigned total_atoms = coords.dimension(0);
      unsigned tidx = ((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x + threadIdx.x;
      float3 *coord_data = (float3*)coords.data();
      float *types = type_index.data();
      float *radii_data = radii.data();
      Dtype *outgrid = out.data();

      //if there are more then LMG_CUDA_NUM_THREADS atoms, chunk them
      for(unsigned atomoffset = 0; atomoffset < total_atoms; atomoffset += LMG_CUDA_NUM_THREADS) {
        //first parallelize over atoms to figure out if they might overlap this block
        unsigned aidx = atomoffset + tidx;
        
        if(aidx < total_atoms && types[aidx] >= 0) {
          atomMask[tidx] = atom_overlaps_block(aidx, grid_origin, gmaker.get_resolution(), coord_data, radii_data[aidx], gmaker.get_radiusmultiple());
        }
        else {
          atomMask[tidx] = 0;
        }

        __syncthreads();
        
        //scan the mask to get just relevant indices
        sharedMemExclusiveScan(tidx, atomMask, scanOutput);
        
        __syncthreads();
        
        //do scatter (stream compaction)
        if(atomMask[tidx])
        {
          atomIndices[scanOutput[tidx]] = tidx + atomoffset;
        }
        __syncthreads();

        unsigned rel_atoms = scanOutput[LMG_CUDA_NUM_THREADS - 1] + atomMask[LMG_CUDA_NUM_THREADS - 1];
        //atomIndex is now a list of rel_atoms possibly relevant atom indices
        gmaker.set_atoms<Dtype, Binary>(rel_atoms, grid_origin, coord_data, types, radii_data, outgrid);

        __syncthreads();//everyone needs to finish before we muck with atomIndices again
      }
    }

    template <typename Dtype>
    void GridMaker::forward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
        Grid<Dtype, 4, true>& out) const {
      //threads are laid out in three dimensions to match the voxel grid, 
      //8x8x8=512 threads per block
      dim3 threads(LMG_CUDA_BLOCKDIM, LMG_CUDA_BLOCKDIM, LMG_CUDA_BLOCKDIM);
      unsigned blocksperside = ceil(dim / float(LMG_CUDA_BLOCKDIM));
      dim3 blocks(blocksperside, blocksperside, blocksperside);
      float3 grid_origin = get_grid_origin(grid_center);

      check_index_args(coords, type_index, radii, out);
      if(radii_type_indexed) {
        throw std::invalid_argument("Type indexed radii not supported with index types.");
      }
      if(blocksperside == 0) {
	throw std::invalid_argument("Zero sized grid.");
      }
      //zero out grid to start
      LMG_CUDA_CHECK(cudaMemset(out.data(), 0, out.size() * sizeof(float)));

      if(coords.dimension(0) == 0) return; //no atoms

      if(binary)
        forward_gpu<Dtype, true><<<blocks, threads>>>(*this, grid_origin, coords, type_index, radii, out);
      else
        forward_gpu<Dtype, false><<<blocks, threads>>>(*this, grid_origin, coords, type_index, radii, out);

      LMG_CUDA_CHECK(cudaPeekAtLastError());
    }

    template void GridMaker::forward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii, Grid<float, 4, true>& out) const;
    template void GridMaker::forward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii, Grid<double, 4, true>& out) const;


    template <typename Dtype, bool Binary, bool RadiiFromTypes>
    __device__ void GridMaker::set_atoms(unsigned rel_atoms, float3 grid_origin,
        const float3 *coord_data, const float *tdata, unsigned ntypes,
        const float *radii, Dtype *data) {
      //figure out what grid point we are
      unsigned xi = threadIdx.x + blockIdx.x * blockDim.x;
      unsigned yi = threadIdx.y + blockIdx.y * blockDim.y;
      unsigned zi = threadIdx.z + blockIdx.z * blockDim.z;

      if(xi >= dim || yi >= dim || zi >= dim)
        return;//bail if we're off-grid, this should not be common

      //compute x,y,z coordinate of grid point
      float3 grid_coords;
      grid_coords.x = xi * resolution + grid_origin.x;
      grid_coords.y = yi * resolution + grid_origin.y;
      grid_coords.z = zi * resolution + grid_origin.z;
      unsigned goffset = ((xi*dim)+yi)*dim + zi; //offset into channel grid
      unsigned chmult = dim*dim*dim; //what to multiply type/channel seletion by

      //iterate over all possibly relevant atoms
      for(unsigned ai = 0; ai < rel_atoms; ai++) {
        unsigned i = atomIndices[ai];
        float3 c = coord_data[i];
        float val = 0;
        if(!RadiiFromTypes) {
          val = calc_point<Binary>(c.x, c.y, c.z, radii[i], grid_coords);
          if(val == 0) continue;
        }

        const float *atom_type_mult = tdata+(ntypes*i); //type vector for this atom
        for(unsigned atype = 0; atype < ntypes; atype++) {
          float tmult = atom_type_mult[atype];
          if(tmult != 0) {
            if(RadiiFromTypes) {
              //need to wait until here to get the right radius
              val = calc_point<Binary>(c.x, c.y, c.z, radii[atype], grid_coords);
              if( val == 0) continue;
            }

            if(Binary) {
              data[atype*chmult+goffset] += tmult;
            } else  {
              data[atype*chmult+goffset] += val*tmult;
            }
          }
        }
      }
    }


    template <typename Dtype, bool Binary, bool RadiiTypeIndexed>
    __global__ void
  //  __launch_bounds__(LMG_CUDA_NUM_THREADS)
    forward_gpu_vec(GridMaker gmaker, float3 grid_origin,
        const Grid<float, 2, true> coords, const Grid<float, 2, true> type_vector,
        const Grid<float, 1, true> radii, float maxradius, Grid<Dtype, 4, true> out) {
      //this is the thread's index within its block, used to parallelize over atoms
      unsigned total_atoms = coords.dimension(0);
      unsigned tidx = ((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x + threadIdx.x; //thread index
      float3 *coord_data = (float3*)coords.data();
      float *types = type_vector.data();
      unsigned ntypes = type_vector.dimension(1);
      float *radii_data = radii.data();
      Dtype *outgrid = out.data();

      //if there are more then LMG_CUDA_NUM_THREADS atoms, chunk them
      for(unsigned atomoffset = 0; atomoffset < total_atoms; atomoffset += LMG_CUDA_NUM_THREADS) {
        //first parallelize over atoms to figure out if they might overlap this block
        unsigned aidx = atomoffset + tidx;

        if(aidx < total_atoms) {
          //assume radii are about the same so can approximate with maxradius
          if(RadiiTypeIndexed)
            atomMask[tidx] = atom_overlaps_block(aidx, grid_origin, gmaker.get_resolution(), coord_data, maxradius, gmaker.get_radiusmultiple());
          else
            atomMask[tidx] = atom_overlaps_block(aidx, grid_origin, gmaker.get_resolution(), coord_data, radii[aidx], gmaker.get_radiusmultiple());
        }
        else {
          atomMask[tidx] = 0;
        }

        __syncthreads();

        //scan the mask to get just relevant indices
        sharedMemExclusiveScan(tidx, atomMask, scanOutput);

        __syncthreads();

        //do scatter (stream compaction)
        if(atomMask[tidx])
        {
          atomIndices[scanOutput[tidx]] = tidx + atomoffset;
        }
        __syncthreads();

        unsigned rel_atoms = scanOutput[LMG_CUDA_NUM_THREADS - 1] + atomMask[LMG_CUDA_NUM_THREADS - 1];
        //atomIndex is now a list of rel_atoms possibly relevant atom indices
        //there should be plenty of parallelism just distributing across grid points, don't bother across types
        gmaker.set_atoms<Dtype, Binary, RadiiTypeIndexed>(rel_atoms, grid_origin, coord_data, types, ntypes, radii_data, outgrid);

        __syncthreads();//everyone needs to finish before we muck with atomIndices again
      }
    }

    template <typename Dtype>
    void GridMaker::forward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 2, true>& type_vector, const Grid<float, 1, true>& radii,
        Grid<Dtype, 4, true>& out) const {

      //threads are laid out in three dimensions to match the voxel grid,
      //8x8x8=512 threads per block
      dim3 threads(LMG_CUDA_BLOCKDIM, LMG_CUDA_BLOCKDIM, LMG_CUDA_BLOCKDIM);
      unsigned blocksperside = ceil(dim / float(LMG_CUDA_BLOCKDIM));
      dim3 blocks(blocksperside, blocksperside, blocksperside);
      float3 grid_origin = get_grid_origin(grid_center);
      unsigned ntypes = type_vector.dimension(1);

      check_vector_args(coords, type_vector, radii, out);
      //zero out grid to start
      LMG_CUDA_CHECK(cudaMemset(out.data(), 0, out.size() * sizeof(float)));

      if(coords.dimension(0) == 0) return; //no atoms

      float maxr = 0;
      if(radii_type_indexed) {
        thrust::device_ptr<float> rptr = thrust::device_pointer_cast(radii.data());
        thrust::device_ptr<float> maxptr = thrust::max_element(rptr, rptr+radii.size());
        maxr = *maxptr;
      }

      if(binary) {
        if(radii_type_indexed)
          forward_gpu_vec<Dtype, true, true><<<blocks, threads>>>(*this, grid_origin, coords, type_vector, radii, maxr, out);
        else
          forward_gpu_vec<Dtype, true, false><<<blocks, threads>>>(*this, grid_origin, coords, type_vector, radii, maxr, out);
      } else {
        if(radii_type_indexed)
          forward_gpu_vec<Dtype, false, true><<<blocks, threads>>>(*this, grid_origin, coords, type_vector, radii, maxr, out);
        else
          forward_gpu_vec<Dtype, false, false><<<blocks, threads>>>(*this, grid_origin, coords, type_vector, radii, maxr, out);
      }

      LMG_CUDA_CHECK(cudaPeekAtLastError());
    }

    template void GridMaker::forward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 2, true>& type_vector, const Grid<float, 1, true>& radii, Grid<float, 4, true>& out) const;
    template void GridMaker::forward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 2, true>& type_vector, const Grid<float, 1, true>& radii, Grid<double, 4, true>& out) const;
 
    
    //batched gpu float
    template void GridMaker::forward<float,2,true>(const Grid<float, 2, true> &centers,
        const Grid<float, 3, true> &coords,
        const Grid<float, 2, true> &types,
        const Grid<float, 2, true> &radii, Grid<float, 5, true> &out) const;
    template void GridMaker::forward<float,3,true>(const Grid<float, 2, true> &centers,
        const Grid<float, 3, true> &coords,
        const Grid<float, 3, true> &types,
        const Grid<float, 2, true> &radii,Grid<float, 5, true> &out) const;

    //batched gpu double
    template void GridMaker::forward<double,2,true>(const Grid<float, 2, true> &centers,
        const Grid<float, 3, true> &coords,
        const Grid<float, 2, true> &types,
        const Grid<float, 2, true> &radii, Grid<double, 5, true> &out) const;
    template void GridMaker::forward<double,3,true>(const Grid<float, 2, true> &centers,
        const Grid<float, 3, true> &coords,
        const Grid<float, 3, true> &types,
        const Grid<float, 2, true> &radii,Grid<double, 5, true> &out) const;
    
    
    //kernel launch - parallelize across whole atoms
    //TODO: accelerate this more
    template<typename Dtype>
    __global__
    void set_atom_gradients(GridMaker G, float3 grid_origin, Grid2fCUDA coords, Grid1fCUDA type_index,
        Grid1fCUDA radii, Grid<Dtype, 4, true> grid, Grid<Dtype, 2, true> atom_gradients) {
      int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if(idx >= type_index.dimension(0)) return;

      //calculate gradient for atom at idx
      float3 agrad{0,0,0};
      float3 a{coords(idx,0),coords(idx,1),coords(idx,2)}; //atom coordinate
      float radius = radii(idx);

      float r = radius * G.radius_scale * G.final_radius_multiple;
      uint2 ranges[3];
      ranges[0] = G.get_bounds_1d(grid_origin.x, a.x, r);
      ranges[1] = G.get_bounds_1d(grid_origin.y, a.y, r);
      ranges[2] = G.get_bounds_1d(grid_origin.z, a.z, r);

      int whichgrid = round(type_index[idx]);
      if(whichgrid < 0) return;
      Grid<Dtype, 3, true> diff = grid[whichgrid];

      //for every grid point possibly overlapped by this atom
      for (unsigned i = ranges[0].x, iend = ranges[0].y; i < iend; ++i) {
        for (unsigned j = ranges[1].x, jend = ranges[1].y; j < jend; ++j) {
          for (unsigned k = ranges[2].x, kend = ranges[2].y; k < kend; ++k) {
            //convert grid point coordinates to angstroms
            float x = grid_origin.x + i * G.resolution;
            float y = grid_origin.y + j * G.resolution;
            float z = grid_origin.z + k * G.resolution;

            G.accumulate_atom_gradient(a.x,a.y,a.z, x,y,z, radius, diff(i,j,k), agrad);
          }
        }
      }
      atom_gradients(idx,0) = agrad.x;
      atom_gradients(idx,1) = agrad.y;
      atom_gradients(idx,2) = agrad.z;
    }

    //type vector version block.y is the type
    template<typename Dtype, bool RadiiFromTypes>
    __global__
    void set_atom_type_gradients(GridMaker G, float3 grid_origin, Grid2fCUDA coords, Grid2fCUDA type_vector,
        unsigned ntypes, Grid1fCUDA radii, Grid<Dtype, 4, true> grid, Grid<Dtype, 2, true> atom_gradients,
        Grid<Dtype, 2, true> type_gradients) {
      int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if(idx >= coords.dimension(0)) return;
      unsigned whicht = blockIdx.y;

      //calculate gradient for atom at idx
      float3 agrad{0,0,0};
      float3 a{coords(idx,0),coords(idx,1),coords(idx,2)}; //atom coordinate
      float radius = 0;
      if(RadiiFromTypes)
        radius = radii(whicht);
      else
        radius = radii(idx);

      float r = radius * G.radius_scale * G.final_radius_multiple;
      uint2 ranges[3];
      ranges[0] = G.get_bounds_1d(grid_origin.x, a.x, r);
      ranges[1] = G.get_bounds_1d(grid_origin.y, a.y, r);
      ranges[2] = G.get_bounds_1d(grid_origin.z, a.z, r);

      Grid<Dtype, 3, true> diff = grid[whicht];

      //for every grid point possibly overlapped by this atom
      float tgrad = 0.0;
      for (unsigned i = ranges[0].x, iend = ranges[0].y; i < iend; ++i) {
        for (unsigned j = ranges[1].x, jend = ranges[1].y; j < jend; ++j) {
          for (unsigned k = ranges[2].x, kend = ranges[2].y; k < kend; ++k) {
            //convert grid point coordinates to angstroms
            float x = grid_origin.x + i * G.resolution;
            float y = grid_origin.y + j * G.resolution;
            float z = grid_origin.z + k * G.resolution;

            G.accumulate_atom_gradient(a.x,a.y,a.z, x,y,z, radius, diff(i,j,k), agrad);

            //type gradient is just some of density vals
            float val;
            if(G.binary)
              val = G.calc_point<true>(a.x, a.y, a.z, radius, float3{x,y,z});
            else
              val = G.calc_point<false>(a.x, a.y, a.z, radius, float3{x,y,z});
            tgrad += val * diff(i,j,k);
          }
        }
      }
      float tmult = type_vector(idx,whicht);
      agrad.x *= tmult;
      agrad.y *= tmult;
      agrad.z *= tmult;

      atomicAdd(&atom_gradients(idx,0), (Dtype)agrad.x);
      atomicAdd(&atom_gradients(idx,1), (Dtype)agrad.y);
      atomicAdd(&atom_gradients(idx,2), (Dtype)agrad.z);

      type_gradients(idx,whicht) = tgrad;
    }



    //gpu accelerated gradient calculation
    template <typename Dtype>
    void GridMaker::backward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
        const Grid<Dtype, 4, true>& grid, Grid<Dtype, 2, true>& atom_gradients) const {
      atom_gradients.fill_zero();
      unsigned n = coords.dimension(0);
      if(n != type_index.size()) throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
      if(n != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");
      if(n != atom_gradients.dimension(0)) throw std::invalid_argument("Gradient dimension doesn't equal number of coordinates");
      if(coords.dimension(1) != 3) throw std::invalid_argument("Coordinates wrong secondary dimension (!= 3)");
      if(radii_type_indexed) {
        throw std::invalid_argument("Type indexed radii not supported with index types.");
      }

      float3 grid_origin = get_grid_origin(grid_center);

      unsigned blocks =  LMG_GET_BLOCKS(n);
      unsigned nthreads = LMG_GET_THREADS(n);
      set_atom_gradients<<<blocks, nthreads>>>(*this, grid_origin, coords, type_index, radii, grid, atom_gradients);
    }

    template void GridMaker::backward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index,const Grid<float, 1, true>& radii,
        const Grid<float, 4, true>& grid, Grid<float, 2, true>& atom_gradients) const;
    template void GridMaker::backward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
        const Grid<double, 4, true>& grid, Grid<double, 2, true>& atom_gradients) const;

    template<typename Dtype>
    void GridMaker::backward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 2, true>& type_vector, const Grid<float, 1, true>& radii,
        const Grid<Dtype, 4, true>& grid,
        Grid<Dtype, 2, true>& atom_gradients, Grid<Dtype, 2, true>& type_gradients) const {
      atom_gradients.fill_zero();
      type_gradients.fill_zero();
      unsigned n = coords.dimension(0);
      unsigned ntypes = type_vector.dimension(1);

      if (n != type_vector.dimension(0)) throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
      if (ntypes != grid.dimension(0)) throw std::invalid_argument("Channels in diff doesn't equal number of types");
      if (n != atom_gradients.dimension(0))
        throw std::invalid_argument("Atom gradient dimension doesn't equal number of coordinates");
      if (n != type_gradients.dimension(0))
        throw std::invalid_argument("Type gradient dimension doesn't equal number of coordinates");
      if (type_gradients.dimension(1) != ntypes)
        throw std::invalid_argument("Type gradient dimension has wrong number of types");
      if (coords.dimension(1) != 3) throw std::invalid_argument("Need x,y,z,r for coord_radius");

      if(radii_type_indexed) { //radii should be size of types
        if(ntypes != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of types");
      } else { //radii should be size of atoms
        if(n != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");
      }

      float3 grid_origin = get_grid_origin(grid_center);


      unsigned blocks = LMG_GET_BLOCKS(n);
      unsigned nthreads = LMG_GET_THREADS(n);
      if(ntypes >= 1024)
        throw std::invalid_argument("Really? More than 1024 types?  The GPU can't handle that.  Are you sure this is a good idea?  I'm giving up.");
      dim3 B(blocks, ntypes, 1); //in theory could support more 1024 by using z, but really..
      if(radii_type_indexed)
        set_atom_type_gradients<Dtype,true><<<B, nthreads>>>(*this, grid_origin, coords, type_vector, ntypes, radii, grid, atom_gradients, type_gradients);
      else
        set_atom_type_gradients<Dtype,false><<<B, nthreads>>>(*this, grid_origin, coords, type_vector, ntypes, radii, grid, atom_gradients, type_gradients);
    }


    template void GridMaker::backward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 2, true>& type_vectors, const Grid<float, 1, true>& radii,
        const Grid<float, 4, true>& grid,
        Grid<float, 2, true>& atom_gradients, Grid<float, 2, true>& type_gradients) const;
    //atomic add doesn't work with double

    //proces grad_grad calculation for a specific atom and type
     template<typename Dtype, bool RadiiFromTypes>
     __global__
     void set_atom_type_grad_grad(GridMaker G, float3 grid_origin, Grid2fCUDA coords, Grid2fCUDA type_vector,
         unsigned ntypes, Grid1fCUDA radii, Grid<Dtype, 4, true> diff, Grid<Dtype, 2, true> atom_gradients,
         Grid<Dtype, 2, true> type_gradients, Grid<Dtype, 4, true> diffdiff, Grid<Dtype, 2, true> atom_diffdiff,
         Grid<Dtype, 2, true> type_diffdiff) {
      int aidx = blockDim.x * blockIdx.x + threadIdx.x;
      if(aidx >= coords.dimension(0)) return;
      unsigned tidx = blockIdx.y;
      Dtype tmult = type_vector(aidx, tidx); //amount of type for this atom
      if(tmult == 0) return;


      float radius = 0;
      if(RadiiFromTypes)
       radius = radii(tidx);
      else
       radius = radii(aidx);

      float ar = radius*G.radius_scale;

      Grid<Dtype, 3, true> diffG = diff[tidx];

      float ax = coords(aidx, 0);
      float ay = coords(aidx, 1);
      float az = coords(aidx, 2);

      float3 agrad; //cartesian gradient
      agrad.x = atom_gradients(aidx, 0);
      agrad.y = atom_gradients(aidx, 1);
      agrad.z = atom_gradients(aidx, 2);

      float tgrad = type_gradients(aidx, tidx);


      float densityr = radius * G.radius_scale * G.final_radius_multiple;
      uint2 bounds[3];
      bounds[0] = G.get_bounds_1d(grid_origin.x, ax, densityr);
      bounds[1] = G.get_bounds_1d(grid_origin.y, ay, densityr);
      bounds[2] = G.get_bounds_1d(grid_origin.z, az, densityr);

      float3 adiffdiff{0,0,0};

      //for every grid point possibly overlapped by this atom
      for (size_t i = bounds[0].x, iend = bounds[0].y; i < iend; i++) {
        for (size_t j = bounds[1].x, jend = bounds[1].y; j < jend; j++) {
          for (size_t k = bounds[2].x, kend = bounds[2].y; k < kend; k++) {
            float x = grid_origin.x + i * G.resolution;
            float y = grid_origin.y + j * G.resolution;
            float z = grid_origin.z + k * G.resolution;
            float Gp = diffG(i,j,k);
            size_t offset = ((((tidx * G.dim) + i) * G.dim) + j) * G.dim + k;

            float dist_x = x - ax;
            float dist_y = y - ay;
            float dist_z = z - az;
            float dist2 = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;
            double dist = sqrt(dist2);

            float agrad_dist = G.density_grad_dist(dist,ar);
            //in backwards did
            // agrad.x += -(dist_x / dist) * (agrad_dist * gridval)
            // differentiate with respect to gridval
            if(isfinite(agrad_dist)) { //overlapping grid position
              float gval = 0.0;
              if(dist > 0) {
                gval += -(dist_x / dist) * (agrad_dist * agrad.x);
                gval += -(dist_y / dist) * (agrad_dist * agrad.y);
                gval += -(dist_z / dist) * (agrad_dist * agrad.z);
                gval *= tmult;
              }

              //type backwards was just the density value
              float val = G.calc_point<false>(ax, ay, az, radius, float3{x,y,z});
              gval += val*tgrad;

              atomicAdd((diffdiff.data() + offset), (Dtype) gval);

              //now accumulate gradient with respect to atom positions
              adiffdiff.x += G.atom_density_grad_grad(ax, x, dist, ar)*Gp*tmult*agrad.x;
              adiffdiff.x += G.atom_density_grad_grad_other(ax, x, ay, y, dist, ar)*Gp*tmult*agrad.y;
              adiffdiff.x += G.atom_density_grad_grad_other(ax, x, az, z, dist, ar)*Gp*tmult*agrad.z;
              adiffdiff.x += G.type_grad_grad(ax, x, dist, ar)*Gp*tgrad;

              adiffdiff.y += G.atom_density_grad_grad_other(ay, y, ax, x, dist, ar)*Gp*tmult*agrad.x;
              adiffdiff.y += G.atom_density_grad_grad(ay, y, dist, ar)*Gp*tmult*agrad.y;
              adiffdiff.y += G.atom_density_grad_grad_other(ay, y, az, z, dist, ar)*Gp*tmult*agrad.z;
              adiffdiff.y += G.type_grad_grad(ay, y, dist, ar)*Gp*tgrad;

              adiffdiff.z += G.atom_density_grad_grad_other(az, z, ax, x, dist, ar)*Gp*tmult*agrad.x;
              adiffdiff.z += G.atom_density_grad_grad_other(az, z, ay, y, dist, ar)*Gp*tmult*agrad.y;
              adiffdiff.z += G.atom_density_grad_grad(az, z, dist, ar)*Gp*tmult*agrad.z;
              adiffdiff.z += G.type_grad_grad(az, z, dist, ar)*Gp*tgrad;
            } //if valid grid point
          } //k
        } //j
      } //i
      atomicAdd(&atom_diffdiff(aidx,0), (Dtype) adiffdiff.x);
      atomicAdd(&atom_diffdiff(aidx,1), (Dtype) adiffdiff.y);
      atomicAdd(&atom_diffdiff(aidx,2), (Dtype) adiffdiff.z);

    }

    template <typename Dtype>
    void GridMaker::backward_gradients(float3 grid_center,  const Grid<float, 2, true>& coords,
        const Grid<float, 2, true>& type_vector, const Grid<float, 1, true>& radii,
        const Grid<Dtype, 4, true>& diff,
        const Grid<Dtype, 2, true>& atom_gradients, const Grid<Dtype, 2, true>& type_gradients,
        Grid<Dtype, 4, true>& diffdiff,
        Grid<Dtype, 2, true>& atom_diffdiff, Grid<Dtype, 2, true>& type_diffdiff) {

      unsigned n = coords.dimension(0);
      unsigned ntypes = type_vector.dimension(1);
      check_vector_args(coords, type_vector, radii, diff);

      if(n != type_vector.dimension(0)) throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
      if(ntypes != diff.dimension(0)) throw std::invalid_argument("Channels in diff doesn't equal number of types");
      if(n != atom_gradients.dimension(0)) throw std::invalid_argument("Atom gradient dimension doesn't equal number of coordinates");
      if(n != type_gradients.dimension(0)) throw std::invalid_argument("Type gradient dimension doesn't equal number of coordinates");
      if(n != atom_diffdiff.dimension(0)) throw std::invalid_argument("Atom gradient gradients dimension doesn't equal number of coordinates");
      if(n != type_diffdiff.dimension(0)) throw std::invalid_argument("Type gradient gradients dimension doesn't equal number of coordinates");

      if(type_gradients.dimension(1) != ntypes) throw std::invalid_argument("Type gradient dimension has wrong number of types");
      if(type_diffdiff.dimension(1) != ntypes) throw std::invalid_argument("Type gradient dimension has wrong number of types");
      if(coords.dimension(1) != 3) throw std::invalid_argument("Need x,y,z,r for coord_radius");

      if(radii_type_indexed) { //radii should be size of types
        if(ntypes != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of types");
      } else { //radii should be size of atoms
        if(n != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");
      }

      if(binary) throw std::invalid_argument("Binary densities not supported");

      atom_diffdiff.fill_zero();
      type_diffdiff.fill_zero(); //note this is the right answer - density is a linear function of amount of type
      diffdiff.fill_zero();

      float3 grid_origin = get_grid_origin(grid_center);

      unsigned blocks = LMG_GET_BLOCKS(n);
      unsigned nthreads = LMG_GET_THREADS(n);
      if(ntypes >= 1024)
        throw std::invalid_argument("Really? More than 1024 types?  The GPU can't handle that.  Are you sure this is a good idea?  I'm giving up.");
      dim3 B(blocks, ntypes, 1); //in theory could support more 1024 by using z, but really..
      if(radii_type_indexed)
        set_atom_type_grad_grad<Dtype,true><<<B, nthreads>>>(*this, grid_origin, coords, type_vector, ntypes, radii,
                    diff, atom_gradients, type_gradients, diffdiff, atom_diffdiff, type_diffdiff);
      else
        set_atom_type_grad_grad<Dtype,false><<<B, nthreads>>>(*this, grid_origin, coords, type_vector, ntypes, radii,
                    diff, atom_gradients, type_gradients, diffdiff, atom_diffdiff, type_diffdiff);
    }

    template void GridMaker::backward_gradients(float3,  const Grid<float, 2, true>&,
        const Grid<float, 2, true>&, const Grid<float, 1, true>&, const Grid<float, 4, true>&,
        const Grid<float, 2, true>&, const Grid<float, 2, true>&, Grid<float, 4, true>&,
        Grid<float, 2, true>&, Grid<float, 2, true>&);
    //atomicadd doesn't work with double

    //derivative of density type grad with respect to coord
    float GridMaker::type_grad_grad(float a, float x, float dist, float r) {
      float ret = 0.0;
      float dist2 = dist*dist;
      if (dist > r * final_radius_multiple) {//no overlap
        return 0;
      }
      else if (dist <= r * gaussian_radius_multiple) {//gaussian derivative
        float r2 = r*r;
        float ex = -2.0 * dist2 / r2;
        ret = 16*(a-x)*(a-x)*exp(ex)/(r2*r2) - 4*exp(ex)/(r2);
      }
      else {//quadratic derivative
        float ax = a-x;
        float ax2 = ax*ax;
        float term1 = -(E+D*dist/r)*ax2/(pow(dist2,1.5)*r);
        float term2 = D*ax2/(dist2*r*r);
        float term3 = (E+D*dist/r)/(dist*r);
        ret = term1+term2+term3;
      }
      return ret;
    }

    //derivative of density_grad_dist - does not include tmult or G; r should include radius_scale
    float GridMaker::atom_density_grad_grad(float a, float x, float dist, float r) {
      float ret = 0.0;
      float dist2 = dist*dist;
      if (dist > r * final_radius_multiple) {//no overlap
        return 0;
      }
      else if (dist <= r * gaussian_radius_multiple) {//gaussian derivative
        float r2 = r*r;
        float ex = -2.0 * dist2 / r2;
        ret = 16*(a-x)*(a-x)*exp(ex)/(r2*r2) - 4*exp(ex)/(r2);
      }
      else {//quadratic derivative
        float ax = a-x;
        float ax2 = ax*ax;
        float term1 = -(E+D*dist/r)*ax2/(pow(dist2,1.5)*r);
        float term2 = D*ax2/(dist2*r*r);
        float term3 = (E+D*dist/r)/(dist*r);
        ret = term1+term2+term3;
      }
      return ret;
    }

    //derivative of desnity_grad_dist - does not include tmult or G
    //a and x are what we are diff'ign with respect to
    float GridMaker::atom_density_grad_grad_other(float a, float x, float b, float y, float dist, float r) {
      float ret = 0.0;
      float dist2 = dist*dist;
      if (dist > r * final_radius_multiple) {//no overlap
        return 0;
      }
      else if (dist <= r * gaussian_radius_multiple) {//gaussian derivative
        float r2 = r*r;
        float ex = -2.0 * dist2 / r2;
        ret = 16*(a-x)*(b-y)*exp(ex)/(r2*r2);
      }
      else {//quadratic derivative
        float ax = a-x;
        float by = b-y;
        float term1 = -(E+D*dist/r)*ax*by/(pow(dist2,1.5)*r);
        float term2 = D*ax*by/(dist2*r*r);
        ret = term1+term2;
      }
      return ret;
    }

    //ar must include radius scale
    float GridMaker::density_grad_dist(float dist, float ar) const {
      float agrad_dist = 0.0;
      float dist2 = dist*dist;
      if (dist > ar * final_radius_multiple) {//no overlap
        return NAN;
      }
      else if (dist <= ar * gaussian_radius_multiple) {//gaussian derivative
        float ex = -2.0 * dist2 / (ar * ar);
        float coef = -4.0 * dist / (ar * ar);
        agrad_dist = coef * exp(ex);
      }
      else {//quadratic derivative
        agrad_dist = (D*dist/ar + E)/ar;
      }
      return agrad_dist;
    }

    void GridMaker::accumulate_atom_gradient(float ax, float ay, float az,
        float x, float y, float z, float ar, float gridval, float3& agrad) const {
      //sum gradient grid values overlapped by the atom times the
      //derivative of the atom density at each grid point
      float dist_x = x - ax;
      float dist_y = y - ay;
      float dist_z = z - az;
      float dist2 = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;
      double dist = sqrt(dist2);
      ar *= radius_scale;

      float agrad_dist = density_grad_dist(dist,ar);

      // d_loss/d_atomx = d_atomdist/d_atomx * d_gridpoint/d_atomdist * d_loss/d_gridpoint
      // sum across all gridpoints
      //dkoes - the negative sign is because we are considering the derivative of the center vs grid
      if(dist > 0 && isfinite(agrad_dist)) {
        agrad.x += -(dist_x / dist) * (agrad_dist * gridval);
        agrad.y += -(dist_y / dist) * (agrad_dist * gridval);
        agrad.z += -(dist_z / dist) * (agrad_dist * gridval);
      }
    }

    //kernel launch - parallelize across whole atoms
    template<typename Dtype>
    __global__
    void set_atom_relevance(GridMaker G, float3 grid_origin, Grid2fCUDA coords, Grid1fCUDA type_index,  Grid1fCUDA radii,
        Grid<Dtype, 4, true> densitygrid, Grid<Dtype, 4, true> diffgrid, Grid<Dtype, 1, true> relevance) {
      int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if(idx >= type_index.dimension(0)) return;
      if(idx >= radii.dimension(0)) return;

      //calculate gradient for atom at idx
      float3 a{coords(idx,0),coords(idx,1),coords(idx,2)}; //atom coordinate
      float radius = radii(idx);

      float r = radius * G.radius_scale * G.final_radius_multiple;
      uint2 ranges[3];
      ranges[0] = G.get_bounds_1d(grid_origin.x, a.x, r);
      ranges[1] = G.get_bounds_1d(grid_origin.y, a.y, r);
      ranges[2] = G.get_bounds_1d(grid_origin.z, a.z, r);

      int whichgrid = round(type_index[idx]);
      if(whichgrid < 0) return;
      Grid<Dtype, 3, true> diff = diffgrid[whichgrid];
      Grid<Dtype, 3, true> density = densitygrid[whichgrid];

      //for every grid point possibly overlapped by this atom
      float ret = 0;
      for (unsigned i = ranges[0].x, iend = ranges[0].y; i < iend; ++i) {
        for (unsigned j = ranges[1].x, jend = ranges[1].y; j < jend; ++j) {
          for (unsigned k = ranges[2].x, kend = ranges[2].y; k < kend; ++k) {
            //convert grid point coordinates to angstroms
            float x = grid_origin.x + i * G.resolution;
            float y = grid_origin.y + j * G.resolution;
            float z = grid_origin.z + k * G.resolution;
            float val = 0;
            if(G.get_binary())
              val = G.calc_point<true>(a.x, a.y, a.z, radius, float3{x,y,z});
            else
              val = G.calc_point<false>(a.x, a.y, a.z, radius, float3{x,y,z});

            if (val > 0) {
              float denseval = density(i,j,k);
              float gridval = diff(i,j,k);
              if(denseval > 0) {
                //weight by contribution to density grid
                ret += gridval*val/denseval;
              } // surely denseval >= val?
            }
          }
        }
      }
      relevance(idx) = ret;
    }

    template <typename Dtype>
    void GridMaker::backward_relevance(float3 grid_center,  const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
        const Grid<Dtype, 4, true>& density, const Grid<Dtype, 4, true>& diff,
        Grid<Dtype, 1, true>& relevance) const {

      relevance.fill_zero();
      unsigned n = coords.dimension(0);
      if(n != type_index.size()) throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
      if(n != relevance.size()) throw std::invalid_argument("Relevance dimension doesn't equal number of coordinates");
      if(n != radii.size()) throw std::invalid_argument("Radii dimension doesn't equal number of coordinates");
      if(coords.dimension(1) != 3) throw std::invalid_argument("Coordinates and radius wrong secondary dimension");

      float3 grid_origin = get_grid_origin(grid_center);

      unsigned blocks =  LMG_GET_BLOCKS(n);
      unsigned nthreads = LMG_GET_THREADS(n);
      set_atom_relevance<<<blocks, nthreads>>>(*this, grid_origin, coords, type_index, radii, density, diff, relevance);
    }

    template void GridMaker::backward_relevance(float3,  const Grid<float, 2, true>&,
        const Grid<float, 1, true>&, const Grid<float, 1, true>&, const Grid<float, 4, true>&,
        const Grid<float, 4, true>&, Grid<float, 1, true>&) const;
    template void GridMaker::backward_relevance(float3,  const Grid<float, 2, true>&,
        const Grid<float, 1, true>&, const Grid<float, 1, true>&, const Grid<double, 4, true>&,
        const Grid<double, 4, true>&, Grid<double, 1, true>& ) const;


} /* namespace libmolgrid */
