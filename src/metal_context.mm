/*
 * metal_context.mm  –  Objective-C++ implementation of MetalContext
 *
 * Responsibilities:
 *   • Obtain the default MTLDevice (Apple Silicon GPU)
 *   • Compile / load the Metal shader library (default.metallib)
 *   • Cache one MTLComputePipelineState per kernel function
 *   • Provide synchronous dispatch helpers used by the rest of the library
 *
 * Memory model:
 *   Apple Silicon has unified memory – the same physical DRAM is accessed by
 *   both CPU and GPU.  We therefore wrap every plain malloc() pointer in a
 *   zero-copy MTLBuffer (MTLResourceStorageModeShared) and pass it straight
 *   to the GPU without any explicit copy.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#include "libmolgrid/metal_context.h"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

// Throw a C++ exception from an NSError.
void throwNSError(NSError* err, const char* context) {
    if (err) {
        std::string msg = std::string(context) + ": " +
                          [err.localizedDescription UTF8String];
        throw std::runtime_error(msg);
    }
}

// Wrap a page-aligned (malloc'd) pointer in a zero-copy MTLBuffer.
// Memory must be allocated with posix_memalign(16384) — see managed_grid.h.
// The buffer does NOT own the memory; caller keeps the original allocation.
id<MTLBuffer> wrapPtr(id<MTLDevice> device, const void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return nil;
    return [device newBufferWithBytesNoCopy:(void*)ptr
                                     length:bytes
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// MetalContext::Impl  –  all Objective-C state lives here
// ---------------------------------------------------------------------------

namespace libmolgrid {

struct MetalContext::Impl {
    id<MTLDevice>       device      = nil;
    id<MTLCommandQueue> queue       = nil;
    id<MTLLibrary>      library     = nil;

    // Cache of pipeline states, keyed by function name.
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    // -----------------------------------------------------------------------
    Impl() {
        device = MTLCreateSystemDefaultDevice();
        if (!device)
            throw std::runtime_error("MetalContext: no Metal device found. "
                                     "Apple Silicon required.");
        queue = [device newCommandQueue];

        // Load the pre-compiled shader library that CMake places next to the
        // executable / library at build time.
        NSError* err = nil;
        NSString* libPath = nil;

        // 1. Try alongside the dynamic library / executable.
        NSBundle* bundle = [NSBundle mainBundle];
        libPath = [bundle pathForResource:@"libmolgrid_shaders" ofType:@"metallib"];

        // 2. Fall back to the build directory (development builds).
        if (!libPath) {
            // Look for default.metallib in the same directory as this .mm was built.
            libPath = @METAL_LIBRARY_PATH; // injected by CMake via -DMETAL_LIBRARY_PATH=...
        }

        if (libPath && [[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
            library = [device newLibraryWithFile:libPath error:&err];
            throwNSError(err, "MetalContext: failed to load metallib");
        } else {
            // Last resort: try the default library (works when embedded via Xcode).
            library = [device newDefaultLibrary];
            if (!library)
                throw std::runtime_error("MetalContext: could not load Metal shader library. "
                                         "Make sure libmolgrid_shaders.metallib is present.");
        }
    }

    // -----------------------------------------------------------------------
    // Obtain (and cache) a pipeline state for a named kernel function.
    id<MTLComputePipelineState> pipeline(const std::string& name) {
        auto it = pipelines.find(name);
        if (it != pipelines.end()) return it->second;

        NSError* err = nil;
        id<MTLFunction> fn = [library newFunctionWithName:@(name.c_str())];
        if (!fn)
            throw std::runtime_error("MetalContext: kernel function '" + name +
                                     "' not found in shader library");
        id<MTLComputePipelineState> ps =
            [device newComputePipelineStateWithFunction:fn error:&err];
        throwNSError(err, ("MetalContext: pipeline for " + name).c_str());
        pipelines[name] = ps;
        return ps;
    }

    // -----------------------------------------------------------------------
    // Generic 1-D dispatch.  N work items, up to 512 per threadgroup.
    void dispatch1D(const std::string& kernelName,
                    NSUInteger N,
                    std::function<void(id<MTLComputeCommandEncoder>)> setArgs)
    {
        if (N == 0) return;
        id<MTLComputePipelineState> ps = pipeline(kernelName);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        setArgs(enc);
        NSUInteger tpg = std::min(ps.maxTotalThreadsPerThreadgroup, (NSUInteger)512);
        NSUInteger groups = (N + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    // -----------------------------------------------------------------------
    // Generic 3-D dispatch.  blocks × threads in X,Y,Z.
    void dispatch3D(const std::string& kernelName,
                    MTLSize blocks, MTLSize threads,
                    std::function<void(id<MTLComputeCommandEncoder>)> setArgs)
    {
        id<MTLComputePipelineState> ps = pipeline(kernelName);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        setArgs(enc);
        [enc dispatchThreadgroups:blocks threadsPerThreadgroup:threads];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
};

// ---------------------------------------------------------------------------
// MetalContext singleton
// ---------------------------------------------------------------------------

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

MetalContext::MetalContext() : impl_(new Impl()) {}
MetalContext::~MetalContext() { delete impl_; }

// ---------------------------------------------------------------------------
// synchronize / max_element
// ---------------------------------------------------------------------------

void MetalContext::synchronize() {
    // All dispatches above are synchronous (waitUntilCompleted).
    // This is a no-op but kept for API completeness.
}

float MetalContext::max_element(const float* data, size_t n) {
    // Unified memory: just scan on CPU – zero extra cost on Apple Silicon.
    if (n == 0) return 0.0f;
    return *std::max_element(data, data + n);
}

// ---------------------------------------------------------------------------
// Transform kernels
// ---------------------------------------------------------------------------

// Struct matching the Metal shader's constant buffer layout.
struct TransformArgs {
    unsigned n;
    float qa, qb, qc, qd;       // quaternion
    float cx, cy, cz;            // center
    float tx, ty, tz;            // translate
};

void MetalContext::transform_forward(unsigned n,
                                     const QuatParams& Q,
                                     Vec3 center, Vec3 translate,
                                     const float* in, float* out,
                                     bool dotranslate)
{
    TransformArgs args {
        n,
        Q.a, Q.b, Q.c, Q.d,
        center.x, center.y, center.z,
        translate.x, translate.y, translate.z
    };
    const std::string kname = dotranslate ? "transform_forward_translate"
                                          : "transform_forward_rotate";
    size_t bytes = (size_t)n * 3 * sizeof(float);
    id<MTLBuffer> bufIn  = wrapPtr(impl_->device, in,  bytes);
    id<MTLBuffer> bufOut = wrapPtr(impl_->device, out, bytes);

    impl_->dispatch1D(kname, n, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:bufIn  offset:0 atIndex:1];
        [enc setBuffer:bufOut offset:0 atIndex:2];
    });
}

void MetalContext::transform_backward(unsigned n,
                                      const QuatParams& invQ,
                                      Vec3 center, Vec3 translate,
                                      const float* in, float* out,
                                      bool dotranslate)
{
    // Backward = first (optionally) untranslate then inverse-rotate.
    // We reuse the same kernels with the inverse quaternion & negated translate.
    if (dotranslate) {
        // Step 1: untranslate
        TransformArgs targs {
            n,
            1,0,0,0,                                   // identity quat
            0,0,0,                                     // center unused
            -translate.x, -translate.y, -translate.z
        };
        size_t bytes = (size_t)n * 3 * sizeof(float);
        id<MTLBuffer> bufIn  = wrapPtr(impl_->device, in,  bytes);
        id<MTLBuffer> bufOut = wrapPtr(impl_->device, out, bytes);
        impl_->dispatch1D("transform_translate_only", n,
                          [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBytes:&targs length:sizeof(targs) atIndex:0];
            [enc setBuffer:bufIn  offset:0 atIndex:1];
            [enc setBuffer:bufOut offset:0 atIndex:2];
        });
        // Step 2: inverse-rotate (in-place: out→out)
        TransformArgs rargs {
            n,
            invQ.a, invQ.b, invQ.c, invQ.d,
            center.x, center.y, center.z,
            0,0,0
        };
        id<MTLBuffer> bufInOut = wrapPtr(impl_->device, out, bytes);
        impl_->dispatch1D("transform_rotate_only", n,
                          [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBytes:&rargs length:sizeof(rargs) atIndex:0];
            [enc setBuffer:bufInOut offset:0 atIndex:1];
            [enc setBuffer:bufInOut offset:0 atIndex:2];
        });
    } else {
        TransformArgs args {
            n,
            invQ.a, invQ.b, invQ.c, invQ.d,
            center.x, center.y, center.z,
            0,0,0
        };
        size_t bytes = (size_t)n * 3 * sizeof(float);
        id<MTLBuffer> bufIn  = wrapPtr(impl_->device, in,  bytes);
        id<MTLBuffer> bufOut = wrapPtr(impl_->device, out, bytes);
        impl_->dispatch1D("transform_rotate_only", n,
                          [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:bufIn  offset:0 atIndex:1];
            [enc setBuffer:bufOut offset:0 atIndex:2];
        });
    }
}

// ---------------------------------------------------------------------------
// CoordinateSet kernels
// ---------------------------------------------------------------------------

struct SumTypesArgs { unsigned natoms; unsigned ntypes; };

void MetalContext::sum_vector_types(const float* types, unsigned natoms,
                                    unsigned ntypes, float* sum)
{
    SumTypesArgs args {natoms, ntypes};
    id<MTLBuffer> bufTypes = wrapPtr(impl_->device, types,
                                     (size_t)natoms * ntypes * sizeof(float));
    id<MTLBuffer> bufSum   = wrapPtr(impl_->device, sum,
                                     (size_t)ntypes * sizeof(float));
    impl_->dispatch1D("sum_vector_types", ntypes,
                      [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&args  length:sizeof(args) atIndex:0];
        [enc setBuffer:bufTypes offset:0 atIndex:1];
        [enc setBuffer:bufSum   offset:0 atIndex:2];
    });
}

void MetalContext::sum_index_types(const float* type_index, unsigned natoms,
                                   unsigned ntypes, float* sum)
{
    SumTypesArgs args {natoms, ntypes};
    id<MTLBuffer> bufIdx = wrapPtr(impl_->device, type_index,
                                   (size_t)natoms * sizeof(float));
    id<MTLBuffer> bufSum = wrapPtr(impl_->device, sum,
                                   (size_t)ntypes * sizeof(float));
    impl_->dispatch1D("sum_index_types", ntypes,
                      [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&args  length:sizeof(args) atIndex:0];
        [enc setBuffer:bufIdx offset:0 atIndex:1];
        [enc setBuffer:bufSum offset:0 atIndex:2];
    });
}

// ---------------------------------------------------------------------------
// GridMaker forward kernels
// ---------------------------------------------------------------------------

struct ForwardArgs {
    GridMakerParams gm;
    float ox, oy, oz;  // grid_origin
    unsigned natoms;
    unsigned ntypes;
    unsigned dim;
};

void MetalContext::forward_index(const GridMakerParams& gm,
                                 Vec3 grid_origin,
                                 const float* coords, unsigned natoms,
                                 const float* type_index,
                                 const float* radii,
                                 float* out, unsigned ntypes, unsigned dim,
                                 bool binary)
{
    ForwardArgs args { gm, grid_origin.x, grid_origin.y, grid_origin.z,
                       natoms, ntypes, dim };

    size_t coordBytes = (size_t)natoms * 3 * sizeof(float);
    size_t atomBytes  = (size_t)natoms * sizeof(float);
    size_t outBytes   = (size_t)ntypes * dim * dim * dim * sizeof(float);

    id<MTLBuffer> bCoords = wrapPtr(impl_->device, coords,     coordBytes);
    id<MTLBuffer> bTypes  = wrapPtr(impl_->device, type_index, atomBytes);
    id<MTLBuffer> bRadii  = wrapPtr(impl_->device, radii,      atomBytes);
    id<MTLBuffer> bOut    = wrapPtr(impl_->device, out,         outBytes);

    const std::string kname = binary ? "forward_index_binary"
                                     : "forward_index_gaussian";

    // 3-D grid of blocks, 8×8×8 threads per block (= 512).
    unsigned bps = (dim + 7) / 8;
    MTLSize blocks  = MTLSizeMake(bps, bps, bps);
    MTLSize threads = MTLSizeMake(8, 8, 8);

    impl_->dispatch3D(kname, blocks, threads,
                      [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&args  length:sizeof(args) atIndex:0];
        [enc setBuffer:bCoords offset:0 atIndex:1];
        [enc setBuffer:bTypes  offset:0 atIndex:2];
        [enc setBuffer:bRadii  offset:0 atIndex:3];
        [enc setBuffer:bOut    offset:0 atIndex:4];
    });
}

void MetalContext::forward_vec(const GridMakerParams& gm,
                               Vec3 grid_origin,
                               const float* coords, unsigned natoms,
                               const float* type_vector, unsigned ntypes,
                               const float* radii, float maxradius,
                               float* out, unsigned dim,
                               bool binary, bool radii_type_indexed)
{
    struct FwdVecArgs {
        GridMakerParams gm;
        float ox, oy, oz;
        unsigned natoms;
        unsigned ntypes;
        unsigned dim;
        float maxradius;
        int radii_type_indexed;
    } args { gm, grid_origin.x, grid_origin.y, grid_origin.z,
             natoms, ntypes, dim, maxradius, (int)radii_type_indexed };

    size_t coordBytes = (size_t)natoms * 3 * sizeof(float);
    size_t typeBytes  = (size_t)natoms * ntypes * sizeof(float);
    size_t radiiBytes = radii_type_indexed ? (size_t)ntypes * sizeof(float)
                                           : (size_t)natoms * sizeof(float);
    size_t outBytes   = (size_t)ntypes * dim * dim * dim * sizeof(float);

    id<MTLBuffer> bCoords = wrapPtr(impl_->device, coords,      coordBytes);
    id<MTLBuffer> bTypes  = wrapPtr(impl_->device, type_vector, typeBytes);
    id<MTLBuffer> bRadii  = wrapPtr(impl_->device, radii,       radiiBytes);
    id<MTLBuffer> bOut    = wrapPtr(impl_->device, out,          outBytes);

    std::string kname;
    if (binary && radii_type_indexed)      kname = "forward_vec_binary_rti";
    else if (binary)                       kname = "forward_vec_binary";
    else if (radii_type_indexed)           kname = "forward_vec_gaussian_rti";
    else                                   kname = "forward_vec_gaussian";

    unsigned bps = (dim + 7) / 8;
    MTLSize blocks  = MTLSizeMake(bps, bps, bps);
    MTLSize threads = MTLSizeMake(8, 8, 8);

    impl_->dispatch3D(kname, blocks, threads,
                      [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&args  length:sizeof(args) atIndex:0];
        [enc setBuffer:bCoords offset:0 atIndex:1];
        [enc setBuffer:bTypes  offset:0 atIndex:2];
        [enc setBuffer:bRadii  offset:0 atIndex:3];
        [enc setBuffer:bOut    offset:0 atIndex:4];
    });
}

// ---------------------------------------------------------------------------
// GridMaker backward kernels
// ---------------------------------------------------------------------------

void MetalContext::backward_index(const GridMakerParams& gm,
                                  Vec3 grid_origin,
                                  const float* coords,
                                  const float* type_index,
                                  const float* radii,
                                  const float* grid,
                                  float* atom_gradients,
                                  unsigned natoms, unsigned ntypes,
                                  unsigned dim)
{
    struct BwdArgs {
        GridMakerParams gm;
        float ox, oy, oz;
        unsigned natoms;
        unsigned ntypes;
        unsigned dim;
    } args { gm, grid_origin.x, grid_origin.y, grid_origin.z,
             natoms, ntypes, dim };

    id<MTLBuffer> bCoords   = wrapPtr(impl_->device, coords,         natoms*3*sizeof(float));
    id<MTLBuffer> bTypes    = wrapPtr(impl_->device, type_index,     natoms*sizeof(float));
    id<MTLBuffer> bRadii    = wrapPtr(impl_->device, radii,          natoms*sizeof(float));
    id<MTLBuffer> bGrid     = wrapPtr(impl_->device, grid,           ntypes*dim*dim*dim*sizeof(float));
    id<MTLBuffer> bGrad     = wrapPtr(impl_->device, atom_gradients, natoms*3*sizeof(float));

    impl_->dispatch1D("set_atom_gradients", natoms,
                      [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&args  length:sizeof(args) atIndex:0];
        [enc setBuffer:bCoords offset:0 atIndex:1];
        [enc setBuffer:bTypes  offset:0 atIndex:2];
        [enc setBuffer:bRadii  offset:0 atIndex:3];
        [enc setBuffer:bGrid   offset:0 atIndex:4];
        [enc setBuffer:bGrad   offset:0 atIndex:5];
    });
}

void MetalContext::backward_vec(const GridMakerParams& gm,
                                Vec3 grid_origin,
                                const float* coords,
                                const float* type_vector, unsigned ntypes,
                                const float* radii,
                                const float* grid,
                                float* atom_gradients,
                                float* type_gradients,
                                unsigned natoms, unsigned dim,
                                bool radii_type_indexed)
{
    struct BwdVecArgs {
        GridMakerParams gm;
        float ox, oy, oz;
        unsigned natoms;
        unsigned ntypes;
        unsigned dim;
        int radii_type_indexed;
    } args { gm, grid_origin.x, grid_origin.y, grid_origin.z,
             natoms, ntypes, dim, (int)radii_type_indexed };

    size_t radiiBytes = radii_type_indexed ? ntypes*sizeof(float)
                                           : natoms*sizeof(float);

    id<MTLBuffer> bCoords    = wrapPtr(impl_->device, coords,          natoms*3*sizeof(float));
    id<MTLBuffer> bTypes     = wrapPtr(impl_->device, type_vector,     natoms*ntypes*sizeof(float));
    id<MTLBuffer> bRadii     = wrapPtr(impl_->device, radii,           radiiBytes);
    id<MTLBuffer> bGrid      = wrapPtr(impl_->device, grid,            ntypes*dim*dim*dim*sizeof(float));
    id<MTLBuffer> bAtomGrad  = wrapPtr(impl_->device, atom_gradients,  natoms*3*sizeof(float));
    id<MTLBuffer> bTypeGrad  = wrapPtr(impl_->device, type_gradients,  natoms*ntypes*sizeof(float));

    const std::string kname = radii_type_indexed ? "set_atom_type_gradients_rti"
                                                 : "set_atom_type_gradients";
    // 2-D block grid: x=atoms, y=types
    unsigned ablocks = (natoms + 511) / 512;
    MTLSize blocks  = MTLSizeMake(ablocks, ntypes, 1);
    MTLSize threads = MTLSizeMake(512, 1, 1);

    impl_->dispatch3D(kname, blocks, threads,
                      [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&args   length:sizeof(args) atIndex:0];
        [enc setBuffer:bCoords    offset:0 atIndex:1];
        [enc setBuffer:bTypes     offset:0 atIndex:2];
        [enc setBuffer:bRadii     offset:0 atIndex:3];
        [enc setBuffer:bGrid      offset:0 atIndex:4];
        [enc setBuffer:bAtomGrad  offset:0 atIndex:5];
        [enc setBuffer:bTypeGrad  offset:0 atIndex:6];
    });
}

void MetalContext::backward_gradients(const GridMakerParams& gm,
                                      Vec3 grid_origin,
                                      const float* coords,
                                      const float* type_vector, unsigned ntypes,
                                      const float* radii,
                                      const float* diff,
                                      const float* atom_gradients,
                                      const float* type_gradients,
                                      float* diffdiff,
                                      float* atom_diffdiff,
                                      float* type_diffdiff,
                                      unsigned natoms, unsigned dim,
                                      bool radii_type_indexed)
{
    struct GGArgs {
        GridMakerParams gm;
        float ox, oy, oz;
        unsigned natoms;
        unsigned ntypes;
        unsigned dim;
        int radii_type_indexed;
    } args { gm, grid_origin.x, grid_origin.y, grid_origin.z,
             natoms, ntypes, dim, (int)radii_type_indexed };

    size_t radiiBytes = radii_type_indexed ? ntypes*sizeof(float) : natoms*sizeof(float);

    id<MTLBuffer> bCoords    = wrapPtr(impl_->device, coords,          natoms*3*sizeof(float));
    id<MTLBuffer> bTypes     = wrapPtr(impl_->device, type_vector,     natoms*ntypes*sizeof(float));
    id<MTLBuffer> bRadii     = wrapPtr(impl_->device, radii,           radiiBytes);
    id<MTLBuffer> bDiff      = wrapPtr(impl_->device, diff,            ntypes*dim*dim*dim*sizeof(float));
    id<MTLBuffer> bAGrad     = wrapPtr(impl_->device, atom_gradients,  natoms*3*sizeof(float));
    id<MTLBuffer> bTGrad     = wrapPtr(impl_->device, type_gradients,  natoms*ntypes*sizeof(float));
    id<MTLBuffer> bDD        = wrapPtr(impl_->device, diffdiff,        ntypes*dim*dim*dim*sizeof(float));
    id<MTLBuffer> bADD       = wrapPtr(impl_->device, atom_diffdiff,   natoms*3*sizeof(float));
    id<MTLBuffer> bTDD       = wrapPtr(impl_->device, type_diffdiff,   natoms*ntypes*sizeof(float));

    const std::string kname = radii_type_indexed ? "set_atom_type_grad_grad_rti"
                                                 : "set_atom_type_grad_grad";
    unsigned ablocks = (natoms + 511) / 512;
    MTLSize blocks  = MTLSizeMake(ablocks, ntypes, 1);
    MTLSize threads = MTLSizeMake(512, 1, 1);

    impl_->dispatch3D(kname, blocks, threads,
                      [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:bCoords offset:0 atIndex:1];
        [enc setBuffer:bTypes  offset:0 atIndex:2];
        [enc setBuffer:bRadii  offset:0 atIndex:3];
        [enc setBuffer:bDiff   offset:0 atIndex:4];
        [enc setBuffer:bAGrad  offset:0 atIndex:5];
        [enc setBuffer:bTGrad  offset:0 atIndex:6];
        [enc setBuffer:bDD     offset:0 atIndex:7];
        [enc setBuffer:bADD    offset:0 atIndex:8];
        [enc setBuffer:bTDD    offset:0 atIndex:9];
    });
}

void MetalContext::backward_relevance(const GridMakerParams& gm,
                                      Vec3 grid_origin,
                                      const float* coords,
                                      const float* type_index,
                                      const float* radii,
                                      const float* density,
                                      const float* diff,
                                      float* relevance,
                                      unsigned natoms, unsigned ntypes,
                                      unsigned dim)
{
    struct RelArgs {
        GridMakerParams gm;
        float ox, oy, oz;
        unsigned natoms;
        unsigned ntypes;
        unsigned dim;
    } args { gm, grid_origin.x, grid_origin.y, grid_origin.z,
             natoms, ntypes, dim };

    id<MTLBuffer> bCoords  = wrapPtr(impl_->device, coords,     natoms*3*sizeof(float));
    id<MTLBuffer> bTypes   = wrapPtr(impl_->device, type_index, natoms*sizeof(float));
    id<MTLBuffer> bRadii   = wrapPtr(impl_->device, radii,      natoms*sizeof(float));
    id<MTLBuffer> bDens    = wrapPtr(impl_->device, density,    ntypes*dim*dim*dim*sizeof(float));
    id<MTLBuffer> bDiff    = wrapPtr(impl_->device, diff,       ntypes*dim*dim*dim*sizeof(float));
    id<MTLBuffer> bRel     = wrapPtr(impl_->device, relevance,  natoms*sizeof(float));

    impl_->dispatch1D("set_atom_relevance", natoms,
                      [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:bCoords offset:0 atIndex:1];
        [enc setBuffer:bTypes  offset:0 atIndex:2];
        [enc setBuffer:bRadii  offset:0 atIndex:3];
        [enc setBuffer:bDens   offset:0 atIndex:4];
        [enc setBuffer:bDiff   offset:0 atIndex:5];
        [enc setBuffer:bRel    offset:0 atIndex:6];
    });
}

// ---------------------------------------------------------------------------
// GridInterpolater kernel
// ---------------------------------------------------------------------------

void MetalContext::grid_interpolate(const float* in,  unsigned in_dim,
                                    Vec3 in_origin,  float in_res,
                                    const QuatParams& invQ,
                                    Vec3 untranslate, Vec3 center,
                                    float* out, unsigned out_dim,
                                    Vec3 out_origin, float out_res)
{
    struct InterpArgs {
        float in_ox, in_oy, in_oz, in_res;
        unsigned in_dim;
        float out_ox, out_oy, out_oz, out_res;
        unsigned out_dim;
        float qa, qb, qc, qd;
        float utx, uty, utz;
        float cx, cy, cz;
    } args {
        in_origin.x, in_origin.y, in_origin.z, in_res, in_dim,
        out_origin.x, out_origin.y, out_origin.z, out_res, out_dim,
        invQ.a, invQ.b, invQ.c, invQ.d,
        untranslate.x, untranslate.y, untranslate.z,
        center.x, center.y, center.z
    };

    id<MTLBuffer> bIn  = wrapPtr(impl_->device, in,  (size_t)in_dim*in_dim*in_dim*sizeof(float));
    id<MTLBuffer> bOut = wrapPtr(impl_->device, out, (size_t)out_dim*out_dim*out_dim*sizeof(float));

    unsigned bps = (out_dim + 7) / 8;
    MTLSize blocks  = MTLSizeMake(bps, bps, bps);
    MTLSize threads = MTLSizeMake(8, 8, 8);

    impl_->dispatch3D("gpu_set_outgrid", blocks, threads,
                      [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:bIn  offset:0 atIndex:1];
        [enc setBuffer:bOut offset:0 atIndex:2];
    });
}

} // namespace libmolgrid
