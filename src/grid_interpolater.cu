/*
 * grid_interpolater.cu
 *
 *  Created on: Aug 6, 2021
 *      Author: dkoes
 */

#include "libmolgrid/grid_interpolater.h"
#include <cmath>
#include <vector>
#include <iomanip>

namespace libmolgrid {

  //wrapper that returns pad if out of bounds
  template <typename Dtype, bool isCUDA>
  CUDA_CALLABLE_MEMBER Dtype GridInterpolater::get_pt(const Grid<Dtype, 3, isCUDA>& in, int x, int y, int z) const {
    if(x < 0 || x >= int(in_dim) || y < 0 || y >= int(in_dim) || z < 0 || z >= int(in_dim))
      return 0;
    else
      return in[x][y][z];
  }

  //given a non-rounded gridpoint in the input grid linearly interpolate values
  template <typename Dtype, bool isCUDA>
  CUDA_CALLABLE_MEMBER Dtype GridInterpolater::interpolate(const Grid<Dtype, 3, isCUDA>& in, float3 gridpt) const {
    //https://en.wikipedia.org/wiki/Trilinear_interpolation
    int xl = floor(gridpt.x);
    int xh = ceil(gridpt.x);
    int yl = floor(gridpt.y);
    int yh = ceil(gridpt.y);
    int zl = floor(gridpt.z);
    int zh = ceil(gridpt.z);

    Dtype p000 = get_pt(in, xl, yl, zl);
    Dtype p001 = get_pt(in, xl, yl, zh);
    Dtype p010 = get_pt(in, xl, yh, zl);
    Dtype p011 = get_pt(in, xl, yh, zh);
    Dtype p100 = get_pt(in, xh, yl, zl);
    Dtype p101 = get_pt(in, xh, yl, zh);
    Dtype p110 = get_pt(in, xh, yh, zl);
    Dtype p111 = get_pt(in, xh, yh, zh);

    Dtype xd = xh > xl ? (gridpt.x-xl)/(xh-xl) : 0;
    Dtype yd = yh > yl ? (gridpt.y-yl)/(yh-yl) : 0;
    Dtype zd = zh > zl ? (gridpt.z-zl)/(zh-zl) : 0;

    Dtype c00 = p000*(1-xd) + p100*xd;
    Dtype c01 = p001*(1-xd) + p101*xd;
    Dtype c10 = p010*(1-xd) + p110*xd;
    Dtype c11 = p011*(1-xd) + p111*xd;

    Dtype c0 = c00*(1-yd)+c10*yd;
    Dtype c1 = c01*(1-yd)+c11*yd;

    Dtype c = c0*(1-zd)+c1*zd;
    return c;
  }

  template float GridInterpolater::interpolate(const Grid<float, 3, true>& in, float3 gridpt) const;
  template float GridInterpolater::interpolate(const Grid<float, 3, false>& in, float3 gridpt) const;
  template double GridInterpolater::interpolate(const Grid<double, 3, false>& in, float3 gridpt) const;

  //convert to texture coords
  __device__ float3 cart2tex(float3 origin, float resolution, float x, float y, float z) {
    //textures interpolate assuming value is in center of pixel instead of at grid point
      float3 pt = { 0.5f+(x-origin.x)/resolution, 0.5f+(y-origin.y)/resolution, 0.5f+(z-origin.z)/resolution };
      return pt;
  }

  //use texture memory to perform interpolation
  __global__ void
  gpu_set_outgrid_texture(cudaTextureObject_t tex,
              float3 in_origin, float in_res, unsigned in_dim,
              float3 out_origin, float out_res, unsigned out_dim,
              Quaternion invQ, float3 untranslate, float3 center,
              Grid<float, 3, true> out) {
    //figure out coordinate we are setting for out
    unsigned xi = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned yi = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned zi = threadIdx.z + blockIdx.z * blockDim.z;

    if(xi >= out_dim || yi >= out_dim || zi >= out_dim)
      return;//bail if we're off-grid, this should not be common

    //compute x,y,z coordinate of grid point
    float3 outpt;
    outpt.x = xi * out_res + out_origin.x;
    outpt.y = yi * out_res + out_origin.y;
    outpt.z = zi * out_res + out_origin.z;

    //apply inverse transformation
    float3 newpt = invQ.rotate(outpt.x+untranslate.x, outpt.y+untranslate.y, outpt.z+untranslate.z);
    //get (not rounded) input grid coordinates (not Cartesian)

    float3 inpt = cart2tex(in_origin, in_res, newpt.x+center.x, newpt.y+center.y, newpt.z+center.z);

    //lookup in normalized texture
    float val = tex3D<float>(tex, inpt.z, inpt.y, inpt.x); //why reverse order?

    //set
    out(xi,yi,zi) = val;
  }

  //interpolate manually
  __global__ void
  gpu_set_outgrid(GridInterpolater interp, Grid<float, 3, true> in,
              float3 in_origin, float in_res, unsigned in_dim,
              float3 out_origin, float out_res, unsigned out_dim,
              Quaternion invQ, float3 untranslate, float3 center,
              Grid<float, 3, true> out) {
    //figure out coordinate we are setting for out
    unsigned xi = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned yi = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned zi = threadIdx.z + blockIdx.z * blockDim.z;

    if(xi >= out_dim || yi >= out_dim || zi >= out_dim)
      return;//bail if we're off-grid, this should not be common

    //compute x,y,z coordinate of grid point
    float3 outpt;
    outpt.x = xi * out_res + out_origin.x;
    outpt.y = yi * out_res + out_origin.y;
    outpt.z = zi * out_res + out_origin.z;

    //apply inverse transformation
    float3 newpt = invQ.rotate(outpt.x+untranslate.x, outpt.y+untranslate.y, outpt.z+untranslate.z);
    //get (not rounded) input grid coordinates (not Cartesian)
    float3 inpt = cart2grid(in_origin, in_res, newpt.x+center.x, newpt.y+center.y, newpt.z+center.z);
    //interpolate
     out(xi,yi,zi) = interp.interpolate(in, inpt);
  }

  template <typename Dtype>
  void GridInterpolater::forward(float3 in_center, const Grid<Dtype, 4, true>& in, const Transform& transform, float3 out_center, Grid<Dtype, 4, true>& out) const {

    checkGrids(in, out);
    float3 center = transform.get_rotation_center();
    float in_radius = in_dimension/2.0;
    float out_radius = out_dimension/2.0;
    float3 in_origin = {in_center.x-in_radius,in_center.y-in_radius,in_center.z-in_radius};
    float3 out_origin = {out_center.x-out_radius,out_center.y-out_radius,out_center.z-out_radius};

    Quaternion invQ = transform.get_quaternion().inverse();
    float3 t = transform.get_translation();
    float3 untranslate = {-t.x-center.x, -t.y-center.y, -t.z-center.z};
    unsigned K = in.dimension(0);

    dim3 threads(LMG_CUDA_BLOCKDIM, LMG_CUDA_BLOCKDIM, LMG_CUDA_BLOCKDIM);
    unsigned blocksperside = ceil(out_dim / float(LMG_CUDA_BLOCKDIM));
    dim3 blocks(blocksperside, blocksperside, blocksperside);

    if(false) {
        //texture memory (mostly) works, but is not faster then interpolating ourself
        //and isn't as close to the cpu version
        //TODO: profile and optimize
        for(unsigned c = 0; c < K; c++) {
            cudaTextureObject_t tex = initializeTexture(in[c]);
            gpu_set_outgrid_texture<<<blocks, threads>>>(tex, in_origin, in_resolution, in_dim, out_origin, out_resolution, out_dim, invQ, untranslate, center, out[c]);
        }
    } else {
      for(unsigned c = 0; c < K; c++) {
          gpu_set_outgrid<<<blocks, threads>>>(*this, in[c], in_origin, in_resolution, in_dim, out_origin, out_resolution, out_dim, invQ, untranslate, center, out[c]);
      }
    }
  }


  template void GridInterpolater::forward(float3 in_center, const Grid<float, 4, true>& in, const Transform& transform, float3 out_center, Grid<float, 4, true>& out) const;


  cudaTextureObject_t GridInterpolater::initializeTexture(const Grid<float, 3, true>& in) const {
    //create an appropriately sized texture memory object for the input
    cudaExtent extent = make_cudaExtent(in_dim, in_dim, in_dim);
    if(!cuArray) { //must allocate array
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
      LMG_CUDA_CHECK(cudaMalloc3DArray(&cuArray, &channelDesc, extent));
    }
    //copy values from in grid
    //first convert to pitched ptr
    cudaPitchedPtr grid = make_cudaPitchedPtr(in.data(), sizeof(float)*in_dim, in_dim, in_dim);
    cudaMemcpy3DParms parms = {0};
    parms.dstArray = cuArray;
    parms.srcPtr = grid;
    parms.extent =  extent;
    parms.kind = cudaMemcpyDeviceToDevice;
    LMG_CUDA_CHECK(cudaMemcpy3D(&parms));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));

    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    LMG_CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    return texObj;
  }

  void GridInterpolater::clearTexture() {
    //deallocate only if allocated
    if(cuArray) {
        cudaFreeArray(cuArray);
        cuArray = nullptr;
    }
  }

}



