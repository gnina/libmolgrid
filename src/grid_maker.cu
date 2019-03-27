#include "grid_maker.h"

namespace libmolgrid {
    __shared__ uint scanScratch[LMG_CUDA_NUM_THREADS * 2];
    __shared__ uint scanBuffer[2][LMG_CUDA_NUM_THREADS];
    __shared__ uint scanOutput[LMG_CUDA_NUM_THREADS];
    __shared__ uint atomIndices[LMG_CUDA_NUM_THREADS];
    __shared__ uint atomMask[LMG_CUDA_NUM_THREADS];

    template <typename Dtype>
    __device__ void zero_grid(Grid<Dtype, 4, true> grid) {
      size_t gsize = grid.size();
      Dtype* gdata = grid.data();
      size_t bIdx = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
      size_t tidx = bIdx * (blockDim.x * blockDim.y * blockDim.z)
                      + (threadIdx.z * (blockDim.x * blockDim.y))
                      + (threadIdx.y * blockDim.x) + threadIdx.x;
      if (tidx < gsize) 
        gdata[tidx] = 0;
    }

    //return squared distance between pt and (x,y,z)
    __device__
    float sqDistance(float coordx, float coordy, float coordz, float x, float y, float z) {
      float ret;
      float tmp = coordx - x;
      ret = tmp * tmp;
      tmp = coordy - y;
      ret += tmp * tmp;
      tmp = coordz - z;
      ret += tmp * tmp;
      return ret;
    }

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
    
    //return 1 if atom potentially overlaps block, 0 otherwise
    __device__
    unsigned GridMaker::atomOverlapsBlock(unsigned aidx, float3 grid_origin, 
        const Grid<float, 2, true>& coords, const Grid<float, 1, true>& type_index, 
        const Grid<float, 1, true>& radii) {
   
      //FIXME: include hydrogens?
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
    
      float r = radii(aidx) * radiusmultiple;
      float centerx = coords(aidx, 0);
      float centery = coords(aidx, 1);
      float centerz = coords(aidx, 2);
    
      //does atom overlap box?
      return !((centerx - r > endx) || (centerx + r < startx)
          || (centery - r > endy) || (centery + r < starty)
          || (centerz - r > endz) || (centerz + r < startz));
    }

    template <typename Dtype>
    __device__ void GridMaker::set_atoms(unsigned rel_atoms, float3& grid_origin, 
        const Grid<float, 2, true>& coords, const Grid<float, 1, true>& type_index, 
        const Grid<float, 1, true>& radii, Grid<Dtype, 4, true>& out) {
      //figure out what grid point we are 
      unsigned xi = threadIdx.x + blockIdx.x * blockDim.x;
      unsigned yi = threadIdx.y + blockIdx.y * blockDim.y;
      unsigned zi = threadIdx.z + blockIdx.z * blockDim.z;

      if(xi >= dim || yi >= dim || zi >= dim)
        return;//bail if we're off-grid, this should not be common

      //compute x,y,z coordinate of grid point
      float x = xi * resolution + grid_origin.x;
      float y = yi * resolution + grid_origin.y;
      float z = zi * resolution + grid_origin.z;

      //iterate over all atoms
      for(unsigned ai = 0; ai < rel_atoms; ai++) {
        unsigned i = atomIndices[ai];
        float coordx = coords(i, 0);
        float coordy = coords(i, 1);
        float coordz = coords(i, 2);
        float d = sqDistance(coordx, coordy, coordz, x, y, z);
        float r = radii(i);
        float atype = type_index(i);
        if (atype >= 0) {    //because of hydrogens on ligands, although 
                             //we already excluded them from the atom list...
          float rsq = r * r;

          if(binary) {
            if(d < rsq) {
              //set gridpoint to 1
              out(atype, xi, yi, zi) = 1.0;
            }
          }
          else {
            //For non-binary density we want a Gaussian where 2 std occurs at the
            //radius, after which it becomes quadratic.  
            //The quadratic is fit to have both the same value and first derivative
            //at the cross over point and a value and derivative of zero at
            //1.5*radius 
            //FIXME wrong for radiusmultiple != 1.5
            float dist = sqrtf(d);
            if (dist < r * radiusmultiple) {
              float h = 0.5 * r;

              if (dist <= r) {
                //return gaussian
                float ex = -dist * dist / (2 * h * h);
                out(atype, xi, yi, zi) += exp(ex);
              }
              else {//return quadratic
                float eval = 1.0 / (M_E * M_E); //e^(-2)
                float q = dist * dist * eval / (h * h) - 6.0 * eval * dist / h + 9.0 * eval;
                out(atype, xi, yi, zi) += q;
              }
            }
          }
        }
      }
    }

    template <typename Dtype>
    __global__ void forward_gpu(GridMaker gmaker, float3 grid_origin,
        const Grid<float, 2, true> coords, const Grid<float, 1, true> type_index, 
        const Grid<float, 1, true> radii, Grid<Dtype, 4, true> out) {
      //zero out memory buffer
      zero_grid(out);

      //this is the thread's index within its block, used to parallelize over atoms
      unsigned total_atoms = radii.size();
      size_t tidx = ((threadIdx.z * blockDim.z) + threadIdx.y) * blockDim.y + threadIdx.x;
      //if there are more then LMG_CUDA_NUM_THREADS atoms, chunk them
      for(unsigned atomoffset = 0; atomoffset < total_atoms; atomoffset += LMG_CUDA_NUM_THREADS) {
        //first parallelize over atoms to figure out if they might overlap this block
        unsigned aidx = atomoffset + tidx;
        
        if(aidx < total_atoms) {
          atomMask[tidx] = gmaker.atomOverlapsBlock(aidx, grid_origin, coords,
              type_index, radii);
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
        gmaker.set_atoms(rel_atoms, grid_origin, coords, type_index, radii, out);
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
      float3 grid_origin = getGridOrigin(grid_center);
      forward_gpu<Dtype><<<blocks, threads>>>(*this, grid_origin, coords, type_index, radii, out);
      LMG_CUDA_CHECK(cudaPeekAtLastError());
    }
} /* namespace libmolgrid */
