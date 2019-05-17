#include "libmolgrid/grid_maker.h"

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
        bounds.y = min(dim, (size_t) ceil(high / resolution));
      }
      return bounds;
    }

    //return 1 if atom potentially overlaps block, 0 otherwise
    __device__
    unsigned GridMaker::atom_overlaps_block(unsigned aidx, float3& grid_origin, 
        const Grid<float, 2, true>& coordrs, const Grid<float, 1, true>& type_index) {
   
      if (type_index(aidx) < 0) return 0; //hydrogen
    
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
    
      float centerx = coordrs(aidx, 0);
      float centery = coordrs(aidx, 1);
      float centerz = coordrs(aidx, 2);
      float r = coordrs(aidx, 3) * radius_scale * final_radius_multiple;
    
      //does atom overlap box?
      return !((centerx - r > endx) || (centerx + r < startx)
          || (centery - r > endy) || (centery + r < starty)
          || (centerz - r > endz) || (centerz + r < startz));
    }

    template <typename Dtype, bool Binary>
    __device__ void GridMaker::set_atoms(size_t rel_atoms, float3& grid_origin, 
        const Grid<float, 2, true>& coords, const Grid<float, 1, true>& type_index, 
        Grid<Dtype, 4, true>& out) {
      //figure out what grid point we are 
      uint3 grid_indices;
      grid_indices.x = threadIdx.x + blockIdx.x * blockDim.x;
      grid_indices.y = threadIdx.y + blockIdx.y * blockDim.y;
      grid_indices.z = threadIdx.z + blockIdx.z * blockDim.z;

      if(grid_indices.x >= dim || grid_indices.y >= dim || grid_indices.z >= dim)
        return;//bail if we're off-grid, this should not be common

      size_t ntypes = out.dimension(0);
      //compute x,y,z coordinate of grid point
      float3 grid_coords;
      grid_coords.x = grid_indices.x * resolution + grid_origin.x;
      grid_coords.y = grid_indices.y * resolution + grid_origin.y;
      grid_coords.z = grid_indices.z * resolution + grid_origin.z;

      //iterate over all possibly relevant atoms
      for(size_t ai = 0; ai < rel_atoms; ai++) {
        size_t i = atomIndices[ai];
        float atype = type_index(i);
        if (atype >= 0 && atype < ntypes) { //should really throw an exception here, but can't
          float3 p{coords(i,0),coords(i,1),coords(i,2)};
          float ar = coords(i,3);
          float val = calc_point(p.x, p.y, p.z, ar, grid_coords);
            if(Binary) {
              if(val != 0) {
                out(atype, grid_indices.x, grid_indices.y, grid_indices.z) = 1.0;
              }
            }
            else {
                // out(atype, grid_indices.x, grid_indices.y, grid_indices.z) += val;
              size_t offset = ((((atype * dim) + grid_indices.x) * dim) +
                  grid_indices.y) * dim + grid_indices.z;
              *(out.data() + offset) += val;
            }
        }
      }
    }

    template <typename Dtype>
    __global__ void forward_gpu(GridMaker gmaker, float3 grid_origin,
        const Grid<float, 2, true> coordrs, const Grid<float, 1, true> type_index,
        Grid<Dtype, 4, true> out) {
      //this is the thread's index within its block, used to parallelize over atoms
      size_t total_atoms = coordrs.dimension(0);
      size_t tidx = ((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x + threadIdx.x;
      //if there are more then LMG_CUDA_NUM_THREADS atoms, chunk them
      for(size_t atomoffset = 0; atomoffset < total_atoms; atomoffset += LMG_CUDA_NUM_THREADS) {
        //first parallelize over atoms to figure out if they might overlap this block
        size_t aidx = atomoffset + tidx;
        
        if(aidx < total_atoms) {
          atomMask[tidx] = gmaker.atom_overlaps_block(aidx, grid_origin, coordrs, type_index);
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

        size_t rel_atoms = scanOutput[LMG_CUDA_NUM_THREADS - 1] + atomMask[LMG_CUDA_NUM_THREADS - 1];
        //atomIndex is now a list of rel_atoms possibly relevant atom indices
        if(gmaker.get_binary())
          gmaker.set_atoms<Dtype,true>(rel_atoms, grid_origin, coordrs, type_index, out);
        else
          gmaker.set_atoms<Dtype,false>(rel_atoms, grid_origin, coordrs, type_index, out);

        __syncthreads();//everyone needs to finish before we muck with atomIndices again
      }
    }

    template <typename Dtype>
    void GridMaker::forward(float3 grid_center, const Grid<float, 2, true>& coordrs,
        const Grid<float, 1, true>& type_index, Grid<Dtype, 4, true>& out) const {
      //threads are laid out in three dimensions to match the voxel grid, 
      //8x8x8=512 threads per block
      dim3 threads(LMG_CUDA_BLOCKDIM, LMG_CUDA_BLOCKDIM, LMG_CUDA_BLOCKDIM);
      unsigned blocksperside = ceil(dim / float(LMG_CUDA_BLOCKDIM));
      dim3 blocks(blocksperside, blocksperside, blocksperside);
      float3 grid_origin = get_grid_origin(grid_center);

      check_index_args(coordrs, type_index, out);
      //zero out grid to start
      LMG_CUDA_CHECK(cudaMemset(out.data(), 0, out.size() * sizeof(float)));

      if(coordrs.dimension(0) == 0) return; //no atoms
      forward_gpu<Dtype><<<blocks, threads>>>(*this, grid_origin, coordrs, type_index, out);
      LMG_CUDA_CHECK(cudaPeekAtLastError());
    }

    template void GridMaker::forward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index, Grid<float, 4, true>& out) const;
    template void GridMaker::forward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index, Grid<double, 4, true>& out) const;

    //kernel launch - parallelize across whole atoms
    //TODO: accelerate this
    template<typename Dtype>
    __global__
    void set_atom_gradients(GridMaker G, float3 grid_origin, Grid2fCUDA coords, Grid1fCUDA type_index,
        Grid<Dtype, 4, true> grid, Grid<Dtype, 2, true> atom_gradients) {
      int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if(idx >= type_index.dimension(0)) return;

      //calculate gradient for atom at idx
      float3 agrad{0,0,0};
      float3 a{coords(idx,0),coords(idx,1),coords(idx,2)}; //atom coordinate
      float radius = coords(idx,3);

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

    //gpu accelerated gradient calculation
    template <typename Dtype>
    void GridMaker::backward(float3 grid_center, const Grid<float, 2, true>& coordrs,
        const Grid<float, 1, true>& type_index, const Grid<Dtype, 4, true>& grid,
        Grid<Dtype, 2, true>& atom_gradients) const {
      atom_gradients.fill_zero();
      unsigned n = coordrs.dimension(0);
      if(n != type_index.size()) throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
      if(n != atom_gradients.dimension(0)) throw std::invalid_argument("Gradient dimension doesn't equal number of coordinates");
      if(coordrs.dimension(1) != 4) throw std::invalid_argument("Coordinates and radius wrong secondary dimension");

      float3 grid_origin = get_grid_origin(grid_center);

      unsigned blocks =  n/LMG_CUDA_NUM_THREADS + bool(n%LMG_CUDA_NUM_THREADS); //at least one if n > 0
      unsigned nthreads = blocks > 1 ? LMG_CUDA_NUM_THREADS : n;
      set_atom_gradients<<<blocks, nthreads>>>(*this, grid_origin, coordrs, type_index, grid, atom_gradients);

    }

    template void GridMaker::backward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index,
        const Grid<float, 4, true>& grid, Grid<float, 2, true>& atom_gradients) const;
    template void GridMaker::backward(float3 grid_center, const Grid<float, 2, true>& coords,
        const Grid<float, 1, true>& type_index,
        const Grid<double, 4, true>& grid, Grid<double, 2, true>& atom_gradients) const;

    float GridMaker::calc_point(float ax, float ay, float az, float ar,
        const float3& grid_coords) const {
      float dx = grid_coords.x - ax;
      float dy = grid_coords.y - ay;
      float dz = grid_coords.z - az;

      float rsq = dx * dx + dy * dy + dz * dz;
      ar *= radius_scale;
      if (binary) {
        //is point within radius?
        if (rsq < ar * ar)
          return 1.0;
        else
          return 0.0;
      } else {
        //For non-binary density we want a Gaussian where 2 std occurs at the
        //radius, after which it becomes quadratic.
        //The quadratic is fit to have both the same value and first derivative
        //at the cross over point and a value and derivative of zero at fianl_radius_multiple
        float dist = sqrtf(rsq);
        if (dist >= ar * final_radius_multiple) {
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
      float agrad_dist = 0.0;
      ar *= radius_scale;
      if (dist >= ar * final_radius_multiple) {//no overlap
        return;
      }
      else if (dist <= ar * gaussian_radius_multiple) {//gaussian derivative
        float ex = -2.0 * dist2 / (ar * ar);
        float coef = -4.0 * dist / (ar * ar);
        agrad_dist = coef * exp(ex);
      }
      else {//quadratic derivative
        agrad_dist = (D*dist/ar + E)/ar;
      }
      // d_loss/d_atomx = d_atomdist/d_atomx * d_gridpoint/d_atomdist * d_loss/d_gridpoint
      // sum across all gridpoints
      //dkoes - the negative sign is because we are considering the derivative of the center vs grid
      if(dist > 0) {
        agrad.x += -(dist_x / dist) * (agrad_dist * gridval);
        agrad.y += -(dist_y / dist) * (agrad_dist * gridval);
        agrad.z += -(dist_z / dist) * (agrad_dist * gridval);
      }
    }

    //kernel launch - parallelize across whole atoms
    template<typename Dtype>
    __global__
    void set_atom_relevance(GridMaker G, float3 grid_origin, Grid2fCUDA coordrs, Grid1fCUDA type_index,
        Grid<Dtype, 4, true> densitygrid, Grid<Dtype, 4, true> diffgrid, Grid<Dtype, 1, true> relevance) {
      int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if(idx >= type_index.dimension(0)) return;

      //calculate gradient for atom at idx
      float3 a{coordrs(idx,0),coordrs(idx,1),coordrs(idx,2)}; //atom coordinate
      float radius = coordrs(idx,3);

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

            float val = G.calc_point(a.x, a.y, a.z, radius, float3{x,y,z});

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
    void GridMaker::backward_relevance(float3 grid_center,  const Grid<float, 2, true>& coordrs,
        const Grid<float, 1, true>& type_index,
        const Grid<Dtype, 4, true>& density, const Grid<Dtype, 4, true>& diff,
        Grid<Dtype, 1, true>& relevance) const {

      relevance.fill_zero();
      unsigned n = coordrs.dimension(0);
      if(n != type_index.size()) throw std::invalid_argument("Type dimension doesn't equal number of coordinates.");
      if(n != relevance.size()) throw std::invalid_argument("Relevance dimension doesn't equal number of coordinates");
      if(coordrs.dimension(1) != 4) throw std::invalid_argument("Coordinates and radius wrong secondary dimension");

      float3 grid_origin = get_grid_origin(grid_center);

      unsigned blocks =  n/LMG_CUDA_NUM_THREADS + bool(n%LMG_CUDA_NUM_THREADS); //at least one if n > 0
      unsigned nthreads = blocks > 1 ? LMG_CUDA_NUM_THREADS : n;
      set_atom_relevance<<<blocks, nthreads>>>(*this, grid_origin, coordrs, type_index, density, diff, relevance);
    }

    template void GridMaker::backward_relevance(float3,  const Grid<float, 2, true>&,
        const Grid<float, 1, true>&, const Grid<float, 4, true>&,
        const Grid<float, 4, true>&, Grid<float, 1, true>&) const;
    template void GridMaker::backward_relevance(float3,  const Grid<float, 2, true>&,
        const Grid<float, 1, true>&, const Grid<double, 4, true>&,
        const Grid<double, 4, true>& , Grid<double, 1, true>& ) const;

} /* namespace libmolgrid */
