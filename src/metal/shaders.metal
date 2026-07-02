/*
 * shaders.metal
 *
 * Metal Shading Language compute kernels for libmolgrid.
 * Translates all CUDA kernels from the original .cu files.
 *
 * Naming conventions:
 *   buffer slot 0  →  constant args struct
 *   buffer slot 1+ →  data buffers (same order as the C++ dispatch wrappers)
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ============================================================================
// Constants (must match common.h)
// ============================================================================

constant uint LMG_BLOCKDIM    = 8;

// ============================================================================
// POD argument structs  (must match metal_context.h / metal_context.mm)
// ============================================================================

struct GridMakerParams {
    float resolution;
    float dimension;
    float radius_scale;
    float gaussian_radius_multiple;
    float final_radius_multiple;
    float A, B, C;
    float D, E;
    uint  dim;
    int   binary;
    int   radii_type_indexed;
};

struct TransformArgs {
    uint  n;
    float qa, qb, qc, qd;
    float cx, cy, cz;
    float tx, ty, tz;
};

struct SumTypesArgs {
    uint natoms;
    uint ntypes;
};

struct ForwardArgs {
    GridMakerParams gm;
    float ox, oy, oz;
    uint  natoms;
    uint  ntypes;
    uint  dim;
};

struct FwdVecArgs {
    GridMakerParams gm;
    float ox, oy, oz;
    uint  natoms;
    uint  ntypes;
    uint  dim;
    float maxradius;
    int   radii_type_indexed;
};

struct BwdArgs {
    GridMakerParams gm;
    float ox, oy, oz;
    uint  natoms;
    uint  ntypes;
    uint  dim;
};

struct BwdVecArgs {
    GridMakerParams gm;
    float ox, oy, oz;
    uint  natoms;
    uint  ntypes;
    uint  dim;
    int   radii_type_indexed;
};

struct GGArgs {
    GridMakerParams gm;
    float ox, oy, oz;
    uint  natoms;
    uint  ntypes;
    uint  dim;
    int   radii_type_indexed;
};

struct RelArgs {
    GridMakerParams gm;
    float ox, oy, oz;
    uint  natoms;
    uint  ntypes;
    uint  dim;
};

struct InterpArgs {
    float in_ox, in_oy, in_oz, in_res;
    uint  in_dim;
    float out_ox, out_oy, out_oz, out_res;
    uint  out_dim;
    float qa, qb, qc, qd;
    float utx, uty, utz;
    float cx, cy, cz;
};

// ============================================================================
// Quaternion helpers
// ============================================================================

static float3 quat_rotate(float qa, float qb, float qc, float qd,
                           float x, float y, float z)
{
    // p' = q * p * q_conj / norm(q)
    float nsq = qa*qa + qb*qb + qc*qc + qd*qd;
    // quaternion multiply: q * (0,x,y,z)
    float ra = -(qb*x + qc*y + qd*z);
    float rb =  qa*x + qc*z - qd*y;
    float rc =  qa*y - qb*z + qd*x;
    float rd =  qa*z + qb*y - qc*x;
    // multiply result by conj(q)/nsq = (qa,-qb,-qc,-qd)/nsq
    float3 out;
    out.x = (-ra*qb + rb*qa - rc*qd + rd*qc) / nsq;
    out.y = (-ra*qc + rb*qd + rc*qa - rd*qb) / nsq;
    out.z = (-ra*qd - rb*qc + rc*qb + rd*qa) / nsq;
    return out;
}

// ============================================================================
// Density helpers (mirror GridMaker::calc_point)
// ============================================================================

static float calc_point_gaussian(float ax, float ay, float az, float ar,
                                  float3 grid_coords,
                                  float radius_scale,
                                  float gaussian_radius_multiple,
                                  float final_radius_multiple,
                                  float A, float B, float C)
{
    ar *= radius_scale;
    float dx = grid_coords.x - ax;
    float dy = grid_coords.y - ay;
    float dz = grid_coords.z - az;
    float rsq = dx*dx + dy*dy + dz*dz;
    float dist = sqrt(rsq);
    if (dist > ar * final_radius_multiple) return 0.0f;
    if (dist <= ar * gaussian_radius_multiple) {
        float ex = -2.0f * rsq / (ar*ar);
        return exp(ex);
    }
    float dr = dist / ar;
    float q  = (A*dr + B)*dr + C;
    return q > 0.0f ? q : 0.0f;
}

static float calc_point_binary(float ax, float ay, float az, float ar,
                                float3 grid_coords, float radius_scale)
{
    ar *= radius_scale;
    float dx = grid_coords.x - ax;
    float dy = grid_coords.y - ay;
    float dz = grid_coords.z - az;
    float rsq = dx*dx + dy*dy + dz*dz;
    return (rsq < ar*ar) ? 1.0f : 0.0f;
}

static uint2 get_bounds_1d(float grid_origin, float coord,
                            float densityrad, float resolution, uint dim)
{
    uint2 bounds = {0u, 0u};
    float low = coord - densityrad - grid_origin;
    if (low > 0) bounds.x = uint(floor(low / resolution));
    float high = coord + densityrad - grid_origin;
    if (high > 0) bounds.y = min(dim, uint(ceil(high / resolution)));
    return bounds;
}

// CAS-based float atomic add (compatible with all Apple GPU families)
static void atomic_add_float(device atomic_uint* addr, float val)
{
    uint old = atomic_load_explicit(addr, memory_order_relaxed);
    uint assumed;
    do {
        assumed = old;
        float updated = as_type<float>(old) + val;
        // Pass &old so Metal updates it with the current value on CAS failure.
        // The loop exits when old == assumed (CAS succeeded).
        atomic_compare_exchange_weak_explicit(addr, &old,
              as_type<uint>(updated),
              memory_order_relaxed, memory_order_relaxed);
    } while (old != assumed);
}

// Check if atom aidx could overlap the current 8×8×8 block (fast AABB cull)
static bool atom_overlaps_block(uint aidx,
                                 float3 grid_origin, float resolution,
                                 const device float* coords,
                                 float radius, float rmult,
                                 uint3 block_id)
{
    float startx = block_id.x * 8 * resolution + grid_origin.x;
    float starty = block_id.y * 8 * resolution + grid_origin.y;
    float startz = block_id.z * 8 * resolution + grid_origin.z;
    float endx = startx + resolution * 8;
    float endy = starty + resolution * 8;
    float endz = startz + resolution * 8;

    float cx_ = coords[aidx*3+0];
    float cy_ = coords[aidx*3+1];
    float cz_ = coords[aidx*3+2];
    float r   = radius * rmult;

    return !((cx_-r > endx) || (cx_+r < startx)
           ||(cy_-r > endy) || (cy_+r < starty)
           ||(cz_-r > endz) || (cz_+r < startz));
}

// ============================================================================
// TRANSFORM KERNELS
// ============================================================================

kernel void transform_forward_translate(
    constant TransformArgs& args [[buffer(0)]],
    const device float* in       [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= args.n) return;
    float x = in[tid*3+0], y = in[tid*3+1], z = in[tid*3+2];
    float3 rot = quat_rotate(args.qa, args.qb, args.qc, args.qd,
                              x - args.cx, y - args.cy, z - args.cz);
    out[tid*3+0] = rot.x + args.cx + args.tx;
    out[tid*3+1] = rot.y + args.cy + args.ty;
    out[tid*3+2] = rot.z + args.cz + args.tz;
}

kernel void transform_forward_rotate(
    constant TransformArgs& args [[buffer(0)]],
    const device float* in       [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= args.n) return;
    float x = in[tid*3+0], y = in[tid*3+1], z = in[tid*3+2];
    float3 rot = quat_rotate(args.qa, args.qb, args.qc, args.qd,
                              x - args.cx, y - args.cy, z - args.cz);
    out[tid*3+0] = rot.x + args.cx;
    out[tid*3+1] = rot.y + args.cy;
    out[tid*3+2] = rot.z + args.cz;
}

kernel void transform_translate_only(
    constant TransformArgs& args [[buffer(0)]],
    const device float* in       [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= args.n) return;
    out[tid*3+0] = in[tid*3+0] + args.tx;
    out[tid*3+1] = in[tid*3+1] + args.ty;
    out[tid*3+2] = in[tid*3+2] + args.tz;
}

kernel void transform_rotate_only(
    constant TransformArgs& args [[buffer(0)]],
    const device float* in       [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= args.n) return;
    float x = in[tid*3+0], y = in[tid*3+1], z = in[tid*3+2];
    float3 rot = quat_rotate(args.qa, args.qb, args.qc, args.qd,
                              x - args.cx, y - args.cy, z - args.cz);
    out[tid*3+0] = rot.x + args.cx;
    out[tid*3+1] = rot.y + args.cy;
    out[tid*3+2] = rot.z + args.cz;
}

// ============================================================================
// COORDINATE SET KERNELS
// ============================================================================

kernel void sum_vector_types(
    constant SumTypesArgs& args  [[buffer(0)]],
    const device float*    types [[buffer(1)]],  // [natoms][ntypes]
    device       float*    sum   [[buffer(2)]],  // [ntypes]
    uint t [[thread_position_in_grid]])
{
    if (t >= args.ntypes) return;
    float tsum = 0.0f;
    for (uint i = 0; i < args.natoms; i++)
        tsum += types[i * args.ntypes + t];
    sum[t] = tsum;
}

kernel void sum_index_types(
    constant SumTypesArgs& args       [[buffer(0)]],
    const device float*    type_index [[buffer(1)]],  // [natoms]
    device       float*    sum        [[buffer(2)]],  // [ntypes]
    uint t [[thread_position_in_grid]])
{
    if (t >= args.ntypes) return;
    float tsum = 0.0f;
    for (uint i = 0; i < args.natoms; i++) {
        if (uint(type_index[i]) == t)
            tsum += 1.0f;
    }
    sum[t] = tsum;
}

// ============================================================================
// GRID MAKER FORWARD  –  Index types
// ============================================================================
// Shared threadgroup arrays declared inside the kernel function.
// Threadgroup size is fixed at 8×8×8 = 512.

// ============================================================================
// GRID MAKER FORWARD  –  Index types
// Each thread owns one voxel and independently loops over all atoms.
// No shared-memory prefix scan needed.
// ============================================================================

kernel void forward_index_binary(
    constant ForwardArgs&  args       [[buffer(0)]],
    const device float*    coords     [[buffer(1)]],  // [natoms][3]
    const device float*    type_index [[buffer(2)]],  // [natoms]
    const device float*    radii      [[buffer(3)]],  // [natoms]
    device       float*    out        [[buffer(4)]],  // [ntypes][dim][dim][dim]
    uint3 tid   [[thread_position_in_threadgroup]],
    uint3 bid   [[threadgroup_position_in_grid]])
{
    uint xi = tid.x + bid.x * 8;
    uint yi = tid.y + bid.y * 8;
    uint zi = tid.z + bid.z * 8;
    uint dim = args.dim;
    if (xi >= dim || yi >= dim || zi >= dim) return;

    float3 go = {args.ox, args.oy, args.oz};
    float3 gc = {xi * args.gm.resolution + go.x,
                 yi * args.gm.resolution + go.y,
                 zi * args.gm.resolution + go.z};
    uint goffset = ((xi * dim) + yi) * dim + zi;
    uint chmult  = dim * dim * dim;

    for (uint i = 0; i < args.natoms; i++) {
        int atype = int(type_index[i]);
        if (atype < 0) continue;
        if (!atom_overlaps_block(i, go, args.gm.resolution, coords,
                                  radii[i], args.gm.final_radius_multiple, bid)) continue;
        float val = calc_point_binary(coords[i*3+0], coords[i*3+1], coords[i*3+2],
                                      radii[i], gc, args.gm.radius_scale);
        if (val != 0.0f)
            out[uint(atype) * chmult + goffset] = 1.0f;
    }
}

kernel void forward_index_gaussian(
    constant ForwardArgs&  args       [[buffer(0)]],
    const device float*    coords     [[buffer(1)]],
    const device float*    type_index [[buffer(2)]],
    const device float*    radii      [[buffer(3)]],
    device       float*    out        [[buffer(4)]],
    uint3 tid   [[thread_position_in_threadgroup]],
    uint3 bid   [[threadgroup_position_in_grid]])
{
    uint xi = tid.x + bid.x * 8;
    uint yi = tid.y + bid.y * 8;
    uint zi = tid.z + bid.z * 8;
    uint dim = args.dim;
    if (xi >= dim || yi >= dim || zi >= dim) return;

    float3 go = {args.ox, args.oy, args.oz};
    float3 gc = {xi * args.gm.resolution + go.x,
                 yi * args.gm.resolution + go.y,
                 zi * args.gm.resolution + go.z};
    uint goffset = ((xi * dim) + yi) * dim + zi;
    uint chmult  = dim * dim * dim;

    for (uint i = 0; i < args.natoms; i++) {
        int atype = int(type_index[i]);
        if (atype < 0) continue;
        if (!atom_overlaps_block(i, go, args.gm.resolution, coords,
                                  radii[i], args.gm.final_radius_multiple, bid)) continue;
        float val = calc_point_gaussian(coords[i*3+0], coords[i*3+1], coords[i*3+2],
                                        radii[i], gc,
                                        args.gm.radius_scale,
                                        args.gm.gaussian_radius_multiple,
                                        args.gm.final_radius_multiple,
                                        args.gm.A, args.gm.B, args.gm.C);
        if (val > 0.0f)
            out[uint(atype) * chmult + goffset] += val;
    }
}


// ============================================================================
// GRID MAKER FORWARD  –  Vector types (4 variants: binary/gaussian × rti/non-rti)
// Each thread owns one voxel and independently loops over all atoms.
// ============================================================================

kernel void forward_vec_binary(
    constant FwdVecArgs&  args       [[buffer(0)]],
    const device float*   coords     [[buffer(1)]],
    const device float*   type_vec   [[buffer(2)]],
    const device float*   radii      [[buffer(3)]],
    device       float*   out        [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]])
{
    uint xi = tid.x + bid.x * 8;
    uint yi = tid.y + bid.y * 8;
    uint zi = tid.z + bid.z * 8;
    uint dim = args.dim;
    if (xi >= dim || yi >= dim || zi >= dim) return;

    float3 go = {args.ox, args.oy, args.oz};
    float3 gc = {xi * args.gm.resolution + go.x,
                 yi * args.gm.resolution + go.y,
                 zi * args.gm.resolution + go.z};
    uint goffset = ((xi * dim) + yi) * dim + zi;
    uint chmult  = dim * dim * dim;

    for (uint i = 0; i < args.natoms; i++) {
        if (!atom_overlaps_block(i, go, args.gm.resolution, coords,
                                  radii[i], args.gm.final_radius_multiple, bid)) continue;
        float ax = coords[i*3+0], ay = coords[i*3+1], az = coords[i*3+2];
        float ar = radii[i];
        float val = calc_point_binary(ax, ay, az, ar, gc, args.gm.radius_scale);
        if (val == 0.0f) continue;
        const device float* atom_types = type_vec + args.ntypes * i;
        for (uint atype = 0; atype < args.ntypes; atype++) {
            float tmult = atom_types[atype];
            if (tmult != 0.0f)
                out[atype * chmult + goffset] += tmult;
        }
    }
}

kernel void forward_vec_gaussian(
    constant FwdVecArgs&  args       [[buffer(0)]],
    const device float*   coords     [[buffer(1)]],
    const device float*   type_vec   [[buffer(2)]],
    const device float*   radii      [[buffer(3)]],
    device       float*   out        [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]])
{
    uint xi = tid.x + bid.x * 8;
    uint yi = tid.y + bid.y * 8;
    uint zi = tid.z + bid.z * 8;
    uint dim = args.dim;
    if (xi >= dim || yi >= dim || zi >= dim) return;

    float3 go = {args.ox, args.oy, args.oz};
    float3 gc = {xi * args.gm.resolution + go.x,
                 yi * args.gm.resolution + go.y,
                 zi * args.gm.resolution + go.z};
    uint goffset = ((xi * dim) + yi) * dim + zi;
    uint chmult  = dim * dim * dim;

    for (uint i = 0; i < args.natoms; i++) {
        if (!atom_overlaps_block(i, go, args.gm.resolution, coords,
                                  radii[i], args.gm.final_radius_multiple, bid)) continue;
        float ax = coords[i*3+0], ay = coords[i*3+1], az = coords[i*3+2];
        float ar = radii[i];
        float val = calc_point_gaussian(ax, ay, az, ar, gc,
                                        args.gm.radius_scale,
                                        args.gm.gaussian_radius_multiple,
                                        args.gm.final_radius_multiple,
                                        args.gm.A, args.gm.B, args.gm.C);
        if (val == 0.0f) continue;
        const device float* atom_types = type_vec + args.ntypes * i;
        for (uint atype = 0; atype < args.ntypes; atype++) {
            float tmult = atom_types[atype];
            if (tmult != 0.0f)
                out[atype * chmult + goffset] += val * tmult;
        }
    }
}

kernel void forward_vec_binary_rti(
    constant FwdVecArgs&  args       [[buffer(0)]],
    const device float*   coords     [[buffer(1)]],
    const device float*   type_vec   [[buffer(2)]],
    const device float*   radii      [[buffer(3)]],  // [ntypes] when rti
    device       float*   out        [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]])
{
    uint xi = tid.x + bid.x * 8;
    uint yi = tid.y + bid.y * 8;
    uint zi = tid.z + bid.z * 8;
    uint dim = args.dim;
    if (xi >= dim || yi >= dim || zi >= dim) return;

    float3 go = {args.ox, args.oy, args.oz};
    float3 gc = {xi * args.gm.resolution + go.x,
                 yi * args.gm.resolution + go.y,
                 zi * args.gm.resolution + go.z};
    uint goffset = ((xi * dim) + yi) * dim + zi;
    uint chmult  = dim * dim * dim;

    for (uint i = 0; i < args.natoms; i++) {
        if (!atom_overlaps_block(i, go, args.gm.resolution, coords,
                                  args.maxradius, args.gm.final_radius_multiple, bid)) continue;
        float ax = coords[i*3+0], ay = coords[i*3+1], az = coords[i*3+2];
        const device float* atom_types = type_vec + args.ntypes * i;
        for (uint atype = 0; atype < args.ntypes; atype++) {
            float tmult = atom_types[atype];
            if (tmult == 0.0f) continue;
            float ar = radii[atype];  // radius-type-indexed
            float val = calc_point_binary(ax, ay, az, ar, gc, args.gm.radius_scale);
            if (val != 0.0f)
                out[atype * chmult + goffset] += tmult;
        }
    }
}

kernel void forward_vec_gaussian_rti(
    constant FwdVecArgs&  args       [[buffer(0)]],
    const device float*   coords     [[buffer(1)]],
    const device float*   type_vec   [[buffer(2)]],
    const device float*   radii      [[buffer(3)]],  // [ntypes] when rti
    device       float*   out        [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]])
{
    uint xi = tid.x + bid.x * 8;
    uint yi = tid.y + bid.y * 8;
    uint zi = tid.z + bid.z * 8;
    uint dim = args.dim;
    if (xi >= dim || yi >= dim || zi >= dim) return;

    float3 go = {args.ox, args.oy, args.oz};
    float3 gc = {xi * args.gm.resolution + go.x,
                 yi * args.gm.resolution + go.y,
                 zi * args.gm.resolution + go.z};
    uint goffset = ((xi * dim) + yi) * dim + zi;
    uint chmult  = dim * dim * dim;

    for (uint i = 0; i < args.natoms; i++) {
        if (!atom_overlaps_block(i, go, args.gm.resolution, coords,
                                  args.maxradius, args.gm.final_radius_multiple, bid)) continue;
        float ax = coords[i*3+0], ay = coords[i*3+1], az = coords[i*3+2];
        const device float* atom_types = type_vec + args.ntypes * i;
        for (uint atype = 0; atype < args.ntypes; atype++) {
            float tmult = atom_types[atype];
            if (tmult == 0.0f) continue;
            float ar = radii[atype];  // radius-type-indexed
            float val = calc_point_gaussian(ax, ay, az, ar, gc,
                                            args.gm.radius_scale,
                                            args.gm.gaussian_radius_multiple,
                                            args.gm.final_radius_multiple,
                                            args.gm.A, args.gm.B, args.gm.C);
            if (val != 0.0f)
                out[atype * chmult + goffset] += val * tmult;
        }
    }
}

// ============================================================================
// GRID MAKER BACKWARD  –  Index types
// ============================================================================

kernel void set_atom_gradients(
    constant BwdArgs&    args          [[buffer(0)]],
    const device float*  coords        [[buffer(1)]],
    const device float*  type_index    [[buffer(2)]],
    const device float*  radii         [[buffer(3)]],
    const device float*  grid          [[buffer(4)]],  // [ntypes][dim][dim][dim]
    device       float*  atom_grads    [[buffer(5)]],  // [natoms][3]
    uint tid [[thread_position_in_grid]])
{
    if (tid >= args.natoms) return;
    int whichgrid = int(round(type_index[tid]));
    if (whichgrid < 0) return;

    float ax = coords[tid*3+0], ay = coords[tid*3+1], az = coords[tid*3+2];
    float radius = radii[tid];
    float ar = radius * args.gm.radius_scale;
    float densityr = ar * args.gm.final_radius_multiple;

    float3 go = {args.ox, args.oy, args.oz};
    uint2 rx = get_bounds_1d(go.x, ax, densityr, args.gm.resolution, args.dim);
    uint2 ry = get_bounds_1d(go.y, ay, densityr, args.gm.resolution, args.dim);
    uint2 rz = get_bounds_1d(go.z, az, densityr, args.gm.resolution, args.dim);

    uint dim = args.dim;
    const device float* diff = grid + (uint)whichgrid * dim*dim*dim;

    float3 agrad = {0,0,0};
    for (uint i = rx.x; i < rx.y; i++) {
        for (uint j = ry.x; j < ry.y; j++) {
            for (uint k = rz.x; k < rz.y; k++) {
                float x = go.x + i*args.gm.resolution;
                float y = go.y + j*args.gm.resolution;
                float z = go.z + k*args.gm.resolution;

                float dx = x-ax, dy = y-ay, dz = z-az;
                float dist2 = dx*dx+dy*dy+dz*dz;
                float dist  = sqrt(dist2);
                if (dist <= 0.0f) continue;

                float agrad_dist = 0.0f;
                if (dist > ar * args.gm.final_radius_multiple) {
                    continue; // no overlap
                } else if (dist <= ar * args.gm.gaussian_radius_multiple) {
                    float ex = -2.0f*dist2/(ar*ar);
                    agrad_dist = (-4.0f*dist/(ar*ar)) * exp(ex);
                } else {
                    agrad_dist = (args.gm.D*dist/ar + args.gm.E)/ar;
                }

                if (!isfinite(agrad_dist)) continue;
                float gridval = diff[(i*dim + j)*dim + k];
                agrad.x += -(dx/dist) * (agrad_dist * gridval);
                agrad.y += -(dy/dist) * (agrad_dist * gridval);
                agrad.z += -(dz/dist) * (agrad_dist * gridval);
            }
        }
    }
    atom_grads[tid*3+0] = agrad.x;
    atom_grads[tid*3+1] = agrad.y;
    atom_grads[tid*3+2] = agrad.z;
}

// ============================================================================
// GRID MAKER BACKWARD  –  Vector types  (2 variants: rti / non-rti)
// ============================================================================

kernel void set_atom_type_gradients(
    constant BwdVecArgs&  args         [[buffer(0)]],
    const device float*   coords       [[buffer(1)]],
    const device float*   type_vec     [[buffer(2)]],  // [natoms][ntypes]
    const device float*   radii        [[buffer(3)]],
    const device float*   grid         [[buffer(4)]],  // [ntypes][dim][dim][dim]
    device       float*   atom_grads   [[buffer(5)]],  // [natoms][3]
    device       float*   type_grads   [[buffer(6)]],  // [natoms][ntypes]
    uint2 gid [[thread_position_in_grid]])
{
    uint idx   = gid.x;   // atom index
    uint whicht= gid.y;   // type index
    if (idx >= args.natoms || whicht >= args.ntypes) return;

    float ax = coords[idx*3+0], ay = coords[idx*3+1], az = coords[idx*3+2];
    float radius = radii[idx]; // non-rti: per-atom
    float ar = radius * args.gm.radius_scale;
    float densityr = ar * args.gm.final_radius_multiple;

    float3 go = {args.ox, args.oy, args.oz};
    uint2 rx = get_bounds_1d(go.x, ax, densityr, args.gm.resolution, args.dim);
    uint2 ry = get_bounds_1d(go.y, ay, densityr, args.gm.resolution, args.dim);
    uint2 rz = get_bounds_1d(go.z, az, densityr, args.gm.resolution, args.dim);

    uint dim = args.dim;
    const device float* diff = grid + whicht*dim*dim*dim;

    float3 agrad = {0,0,0};
    float tgrad  = 0.0f;

    for (uint i = rx.x; i < rx.y; i++) {
        for (uint j = ry.x; j < ry.y; j++) {
            for (uint k = rz.x; k < rz.y; k++) {
                float x = go.x + i*args.gm.resolution;
                float y = go.y + j*args.gm.resolution;
                float z = go.z + k*args.gm.resolution;
                float3 gc = {x,y,z};
                float dx = x-ax, dy = y-ay, dz = z-az;
                float dist2 = dx*dx+dy*dy+dz*dz;
                float dist  = sqrt(dist2);

                float agrad_dist = 0.0f;
                bool finite_ag = false;
                if (dist > 0.0f && dist <= ar * args.gm.final_radius_multiple) {
                    if (dist <= ar * args.gm.gaussian_radius_multiple) {
                        float ex = -2.0f*dist2/(ar*ar);
                        agrad_dist = (-4.0f*dist/(ar*ar))*exp(ex);
                    } else {
                        agrad_dist = (args.gm.D*dist/ar + args.gm.E)/ar;
                    }
                    finite_ag = isfinite(agrad_dist);
                }

                float val;
                if (args.gm.binary)
                    val = calc_point_binary(ax,ay,az,radius,gc,args.gm.radius_scale);
                else
                    val = calc_point_gaussian(ax,ay,az,radius,gc,
                                              args.gm.radius_scale,
                                              args.gm.gaussian_radius_multiple,
                                              args.gm.final_radius_multiple,
                                              args.gm.A,args.gm.B,args.gm.C);

                float gv = diff[(i*dim+j)*dim+k];
                tgrad += val * gv;

                if (finite_ag && dist > 0.0f) {
                    agrad.x += -(dx/dist)*(agrad_dist*gv);
                    agrad.y += -(dy/dist)*(agrad_dist*gv);
                    agrad.z += -(dz/dist)*(agrad_dist*gv);
                }
            }
        }
    }

    float tmult = type_vec[idx*args.ntypes + whicht];
    agrad.x *= tmult; agrad.y *= tmult; agrad.z *= tmult;

    atomic_add_float((device atomic_uint*)(atom_grads + idx*3+0), agrad.x);
    atomic_add_float((device atomic_uint*)(atom_grads + idx*3+1), agrad.y);
    atomic_add_float((device atomic_uint*)(atom_grads + idx*3+2), agrad.z);
    type_grads[idx*args.ntypes + whicht] = tgrad;
}

kernel void set_atom_type_gradients_rti(
    constant BwdVecArgs&  args         [[buffer(0)]],
    const device float*   coords       [[buffer(1)]],
    const device float*   type_vec     [[buffer(2)]],
    const device float*   radii        [[buffer(3)]],  // [ntypes] for rti
    const device float*   grid         [[buffer(4)]],
    device       float*   atom_grads   [[buffer(5)]],
    device       float*   type_grads   [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint idx   = gid.x;
    uint whicht= gid.y;
    if (idx >= args.natoms || whicht >= args.ntypes) return;

    float ax = coords[idx*3+0], ay = coords[idx*3+1], az = coords[idx*3+2];
    float radius = radii[whicht]; // rti: per-type radius
    float ar = radius * args.gm.radius_scale;
    float densityr = ar * args.gm.final_radius_multiple;

    float3 go = {args.ox, args.oy, args.oz};
    uint2 rx = get_bounds_1d(go.x, ax, densityr, args.gm.resolution, args.dim);
    uint2 ry = get_bounds_1d(go.y, ay, densityr, args.gm.resolution, args.dim);
    uint2 rz = get_bounds_1d(go.z, az, densityr, args.gm.resolution, args.dim);

    uint dim = args.dim;
    const device float* diff = grid + whicht*dim*dim*dim;

    float3 agrad = {0,0,0};
    float tgrad  = 0.0f;

    for (uint i = rx.x; i < rx.y; i++) {
        for (uint j = ry.x; j < ry.y; j++) {
            for (uint k = rz.x; k < rz.y; k++) {
                float x = go.x + i*args.gm.resolution;
                float y = go.y + j*args.gm.resolution;
                float z = go.z + k*args.gm.resolution;
                float3 gc = {x,y,z};
                float dx = x-ax, dy = y-ay, dz = z-az;
                float dist2 = dx*dx+dy*dy+dz*dz;
                float dist  = sqrt(dist2);

                float agrad_dist = 0.0f;
                bool finite_ag = false;
                if (dist > 0.0f && dist <= ar * args.gm.final_radius_multiple) {
                    if (dist <= ar * args.gm.gaussian_radius_multiple) {
                        float ex = -2.0f*dist2/(ar*ar);
                        agrad_dist = (-4.0f*dist/(ar*ar))*exp(ex);
                    } else {
                        agrad_dist = (args.gm.D*dist/ar + args.gm.E)/ar;
                    }
                    finite_ag = isfinite(agrad_dist);
                }

                float val;
                if (args.gm.binary)
                    val = calc_point_binary(ax,ay,az,radius,gc,args.gm.radius_scale);
                else
                    val = calc_point_gaussian(ax,ay,az,radius,gc,
                                              args.gm.radius_scale,
                                              args.gm.gaussian_radius_multiple,
                                              args.gm.final_radius_multiple,
                                              args.gm.A,args.gm.B,args.gm.C);
                float gv = diff[(i*dim+j)*dim+k];
                tgrad += val * gv;

                if (finite_ag && dist > 0.0f) {
                    agrad.x += -(dx/dist)*(agrad_dist*gv);
                    agrad.y += -(dy/dist)*(agrad_dist*gv);
                    agrad.z += -(dz/dist)*(agrad_dist*gv);
                }
            }
        }
    }

    float tmult = type_vec[idx*args.ntypes + whicht];
    agrad.x *= tmult; agrad.y *= tmult; agrad.z *= tmult;

    atomic_add_float((device atomic_uint*)(atom_grads + idx*3+0), agrad.x);
    atomic_add_float((device atomic_uint*)(atom_grads + idx*3+1), agrad.y);
    atomic_add_float((device atomic_uint*)(atom_grads + idx*3+2), agrad.z);
    type_grads[idx*args.ntypes + whicht] = tgrad;
}

// ============================================================================
// GRID MAKER BACKWARD_GRADIENTS (grad-of-grad)
// ============================================================================

kernel void set_atom_type_grad_grad(
    constant GGArgs&      args         [[buffer(0)]],
    const device float*   coords       [[buffer(1)]],
    const device float*   type_vec     [[buffer(2)]],
    const device float*   radii        [[buffer(3)]],
    const device float*   diff         [[buffer(4)]],
    const device float*   atom_grads   [[buffer(5)]],
    const device float*   type_grads   [[buffer(6)]],
    device       float*   diffdiff     [[buffer(7)]],
    device       float*   atom_diffdiff[[buffer(8)]],
    device       float*   type_diffdiff[[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint aidx  = gid.x;
    uint tidx  = gid.y;
    if (aidx >= args.natoms || tidx >= args.ntypes) return;

    float tmult = type_vec[aidx*args.ntypes + tidx];
    if (tmult == 0.0f) return;

    float radius = args.radii_type_indexed ? radii[tidx] : radii[aidx];
    float ar = radius * args.gm.radius_scale;

    float ax = coords[aidx*3+0], ay = coords[aidx*3+1], az = coords[aidx*3+2];
    float3 agrad = { atom_grads[aidx*3+0], atom_grads[aidx*3+1], atom_grads[aidx*3+2] };
    float tgrad  = type_grads[aidx*args.ntypes + tidx];

    float densityr = ar * args.gm.final_radius_multiple;
    float3 go = {args.ox, args.oy, args.oz};
    uint2 rx = get_bounds_1d(go.x, ax, densityr, args.gm.resolution, args.dim);
    uint2 ry = get_bounds_1d(go.y, ay, densityr, args.gm.resolution, args.dim);
    uint2 rz = get_bounds_1d(go.z, az, densityr, args.gm.resolution, args.dim);

    uint dim = args.dim;
    const device float* diffG = diff + tidx*dim*dim*dim;
    float3 adiffdiff = {0,0,0};

    for (uint i = rx.x; i < rx.y; i++) {
        for (uint j = ry.x; j < ry.y; j++) {
            for (uint k = rz.x; k < rz.y; k++) {
                float x = go.x + i*args.gm.resolution;
                float y = go.y + j*args.gm.resolution;
                float z = go.z + k*args.gm.resolution;
                float dx = x-ax, dy = y-ay, dz = z-az;
                float dist2 = dx*dx+dy*dy+dz*dz;
                float dist  = sqrt(dist2);

                float Gp = diffG[(i*dim+j)*dim+k];
                uint  offset = ((tidx*dim+i)*dim+j)*dim+k;

                float agrad_dist = 0.0f;
                if (dist > ar * args.gm.final_radius_multiple) continue;
                if (dist <= ar * args.gm.gaussian_radius_multiple) {
                    float ex = -2.0f*dist2/(ar*ar);
                    agrad_dist = (-4.0f*dist/(ar*ar))*exp(ex);
                } else {
                    agrad_dist = (args.gm.D*dist/ar + args.gm.E)/ar;
                }
                if (!isfinite(agrad_dist)) continue;

                float3 gc = {x,y,z};
                float val = calc_point_gaussian(ax,ay,az,radius,gc,
                                                args.gm.radius_scale,
                                                args.gm.gaussian_radius_multiple,
                                                args.gm.final_radius_multiple,
                                                args.gm.A,args.gm.B,args.gm.C);

                float gval = 0.0f;
                if (dist > 0.0f) {
                    gval += -(dx/dist)*(agrad_dist*agrad.x);
                    gval += -(dy/dist)*(agrad_dist*agrad.y);
                    gval += -(dz/dist)*(agrad_dist*agrad.z);
                    gval *= tmult;
                }
                gval += val * tgrad;
                atomic_add_float((device atomic_uint*)(diffdiff + offset), gval);

                // Accumulate full second-order gradient w.r.t. atom positions.
                // Uses the 3×3 Hessian of density w.r.t. atom coords:
                //   d²f/dxa² (diagonal ggd_xx), d²f/dxadxb (cross ggd_xy etc.)
                // Matches CPU: atom_density_grad_grad / type_grad_grad helpers.
                if (dist > 0.0f) {
                    float r2 = ar*ar, d2 = dist2;
                    float ex2 = -2.0f*d2/r2;
                    float ggd_xx, ggd_yy, ggd_zz, ggd_xy, ggd_xz, ggd_yz;
                    if (dist <= ar * args.gm.gaussian_radius_multiple) {
                        float e = exp(ex2);
                        float c = 16.0f*e/(r2*r2);
                        float dc = -4.0f*e/r2;
                        ggd_xx = c*dx*dx + dc;
                        ggd_yy = c*dy*dy + dc;
                        ggd_zz = c*dz*dz + dc;
                        ggd_xy = c*dx*dy;
                        ggd_xz = c*dx*dz;
                        ggd_yz = c*dy*dz;
                    } else {
                        float p = -(args.gm.E + args.gm.D*dist/ar) / (pow(d2, 1.5f) * ar);
                        float q = args.gm.D / (d2 * ar * ar);
                        float s = (args.gm.E + args.gm.D*dist/ar) / (dist * ar);
                        float pq = p + q;
                        ggd_xx = pq*dx*dx + s;
                        ggd_yy = pq*dy*dy + s;
                        ggd_zz = pq*dz*dz + s;
                        ggd_xy = pq*dx*dy;
                        ggd_xz = pq*dx*dz;
                        ggd_yz = pq*dy*dz;
                    }
                    float GptG = Gp * tmult;
                    adiffdiff.x += (ggd_xx*agrad.x + ggd_xy*agrad.y + ggd_xz*agrad.z)*GptG
                                 + ggd_xx*Gp*tgrad;
                    adiffdiff.y += (ggd_xy*agrad.x + ggd_yy*agrad.y + ggd_yz*agrad.z)*GptG
                                 + ggd_yy*Gp*tgrad;
                    adiffdiff.z += (ggd_xz*agrad.x + ggd_yz*agrad.y + ggd_zz*agrad.z)*GptG
                                 + ggd_zz*Gp*tgrad;
                }
            }
        }
    }
    atomic_add_float((device atomic_uint*)(atom_diffdiff + aidx*3+0), adiffdiff.x);
    atomic_add_float((device atomic_uint*)(atom_diffdiff + aidx*3+1), adiffdiff.y);
    atomic_add_float((device atomic_uint*)(atom_diffdiff + aidx*3+2), adiffdiff.z);
}

kernel void set_atom_type_grad_grad_rti(
    constant GGArgs&      args         [[buffer(0)]],
    const device float*   coords       [[buffer(1)]],
    const device float*   type_vec     [[buffer(2)]],
    const device float*   radii        [[buffer(3)]],
    const device float*   diff         [[buffer(4)]],
    const device float*   atom_grads   [[buffer(5)]],
    const device float*   type_grads   [[buffer(6)]],
    device       float*   diffdiff     [[buffer(7)]],
    device       float*   atom_diffdiff[[buffer(8)]],
    device       float*   type_diffdiff[[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Same as above but radius = radii[tidx]
    uint aidx  = gid.x;
    uint tidx  = gid.y;
    if (aidx >= args.natoms || tidx >= args.ntypes) return;

    float tmult = type_vec[aidx*args.ntypes + tidx];
    if (tmult == 0.0f) return;

    float radius = radii[tidx]; // rti
    float ar = radius * args.gm.radius_scale;

    float ax = coords[aidx*3+0], ay = coords[aidx*3+1], az = coords[aidx*3+2];
    float3 agrad = { atom_grads[aidx*3+0], atom_grads[aidx*3+1], atom_grads[aidx*3+2] };
    float tgrad  = type_grads[aidx*args.ntypes + tidx];

    float densityr = ar * args.gm.final_radius_multiple;
    float3 go = {args.ox, args.oy, args.oz};
    uint2 rx = get_bounds_1d(go.x, ax, densityr, args.gm.resolution, args.dim);
    uint2 ry = get_bounds_1d(go.y, ay, densityr, args.gm.resolution, args.dim);
    uint2 rz = get_bounds_1d(go.z, az, densityr, args.gm.resolution, args.dim);

    uint dim = args.dim;
    const device float* diffG = diff + tidx*dim*dim*dim;
    float3 adiffdiff = {0,0,0};

    for (uint i = rx.x; i < rx.y; i++) {
        for (uint j = ry.x; j < ry.y; j++) {
            for (uint k = rz.x; k < rz.y; k++) {
                float x = go.x + i*args.gm.resolution;
                float y = go.y + j*args.gm.resolution;
                float z = go.z + k*args.gm.resolution;
                float dx = x-ax, dy = y-ay, dz = z-az;
                float dist2 = dx*dx+dy*dy+dz*dz;
                float dist  = sqrt(dist2);
                if (dist <= 0.0f || dist > ar*args.gm.final_radius_multiple) continue;

                float Gp = diffG[(i*dim+j)*dim+k];
                uint  offset = ((tidx*dim+i)*dim+j)*dim+k;

                float3 gc = {x,y,z};
                float val = calc_point_gaussian(ax,ay,az,radius,gc,
                                                args.gm.radius_scale,
                                                args.gm.gaussian_radius_multiple,
                                                args.gm.final_radius_multiple,
                                                args.gm.A,args.gm.B,args.gm.C);
                float agrad_dist;
                if (dist <= ar * args.gm.gaussian_radius_multiple) {
                    float ex = -2.0f*dist2/(ar*ar);
                    agrad_dist = (-4.0f*dist/(ar*ar))*exp(ex);
                } else {
                    agrad_dist = (args.gm.D*dist/ar + args.gm.E)/ar;
                }
                if (!isfinite(agrad_dist)) continue;

                float gval = -(dx/dist)*(agrad_dist*agrad.x)*tmult
                           + -(dy/dist)*(agrad_dist*agrad.y)*tmult
                           + -(dz/dist)*(agrad_dist*agrad.z)*tmult
                           + val*tgrad;
                atomic_add_float((device atomic_uint*)(diffdiff + offset), gval);

                // Full second-order gradient (same 3×3 Hessian as non-RTI kernel)
                float r2 = ar*ar, d2 = dist2;
                float ex2 = -2.0f*d2/r2;
                float ggd_xx, ggd_yy, ggd_zz, ggd_xy, ggd_xz, ggd_yz;
                if (dist <= ar * args.gm.gaussian_radius_multiple) {
                    float e = exp(ex2);
                    float c = 16.0f*e/(r2*r2);
                    float dc = -4.0f*e/r2;
                    ggd_xx = c*dx*dx + dc;
                    ggd_yy = c*dy*dy + dc;
                    ggd_zz = c*dz*dz + dc;
                    ggd_xy = c*dx*dy;
                    ggd_xz = c*dx*dz;
                    ggd_yz = c*dy*dz;
                } else {
                    float p = -(args.gm.E + args.gm.D*dist/ar) / (pow(d2, 1.5f) * ar);
                    float q = args.gm.D / (d2 * ar * ar);
                    float s = (args.gm.E + args.gm.D*dist/ar) / (dist * ar);
                    float pq = p + q;
                    ggd_xx = pq*dx*dx + s;
                    ggd_yy = pq*dy*dy + s;
                    ggd_zz = pq*dz*dz + s;
                    ggd_xy = pq*dx*dy;
                    ggd_xz = pq*dx*dz;
                    ggd_yz = pq*dy*dz;
                }
                float GptG = Gp * tmult;
                adiffdiff.x += (ggd_xx*agrad.x + ggd_xy*agrad.y + ggd_xz*agrad.z)*GptG
                             + ggd_xx*Gp*tgrad;
                adiffdiff.y += (ggd_xy*agrad.x + ggd_yy*agrad.y + ggd_yz*agrad.z)*GptG
                             + ggd_yy*Gp*tgrad;
                adiffdiff.z += (ggd_xz*agrad.x + ggd_yz*agrad.y + ggd_zz*agrad.z)*GptG
                             + ggd_zz*Gp*tgrad;
            }
        }
    }
    atomic_add_float((device atomic_uint*)(atom_diffdiff + aidx*3+0), adiffdiff.x);
    atomic_add_float((device atomic_uint*)(atom_diffdiff + aidx*3+1), adiffdiff.y);
    atomic_add_float((device atomic_uint*)(atom_diffdiff + aidx*3+2), adiffdiff.z);
}

// ============================================================================
// GRID MAKER BACKWARD_RELEVANCE
// ============================================================================

kernel void set_atom_relevance(
    constant RelArgs&    args       [[buffer(0)]],
    const device float*  coords     [[buffer(1)]],
    const device float*  type_index [[buffer(2)]],
    const device float*  radii      [[buffer(3)]],
    const device float*  density    [[buffer(4)]],  // [ntypes][dim][dim][dim]
    const device float*  diff       [[buffer(5)]],
    device       float*  relevance  [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= args.natoms) return;
    int whichgrid = int(round(type_index[tid]));
    if (whichgrid < 0) return;

    float ax = coords[tid*3+0], ay = coords[tid*3+1], az = coords[tid*3+2];
    float radius = radii[tid];

    float densityr = radius * args.gm.radius_scale * args.gm.final_radius_multiple;
    float3 go = {args.ox, args.oy, args.oz};
    uint2 rx = get_bounds_1d(go.x, ax, densityr, args.gm.resolution, args.dim);
    uint2 ry = get_bounds_1d(go.y, ay, densityr, args.gm.resolution, args.dim);
    uint2 rz = get_bounds_1d(go.z, az, densityr, args.gm.resolution, args.dim);

    uint dim = args.dim;
    uint gidx = (uint)whichgrid;
    const device float* dens_ch = density + gidx*dim*dim*dim;
    const device float* diff_ch = diff    + gidx*dim*dim*dim;

    float ret = 0.0f;
    for (uint i = rx.x; i < rx.y; i++) {
        for (uint j = ry.x; j < ry.y; j++) {
            for (uint k = rz.x; k < rz.y; k++) {
                float x = go.x + i*args.gm.resolution;
                float y = go.y + j*args.gm.resolution;
                float z = go.z + k*args.gm.resolution;
                float3 gc = {x,y,z};
                float val;
                if (args.gm.binary)
                    val = calc_point_binary(ax,ay,az,radius,gc,args.gm.radius_scale);
                else
                    val = calc_point_gaussian(ax,ay,az,radius,gc,
                                              args.gm.radius_scale,
                                              args.gm.gaussian_radius_multiple,
                                              args.gm.final_radius_multiple,
                                              args.gm.A,args.gm.B,args.gm.C);
                if (val > 0.0f) {
                    float denseval = dens_ch[(i*dim+j)*dim+k];
                    float gridval  = diff_ch[(i*dim+j)*dim+k];
                    if (denseval > 0.0f)
                        ret += gridval * val / denseval;
                }
            }
        }
    }
    relevance[tid] = ret;
}

// ============================================================================
// GRID INTERPOLATER
// ============================================================================

kernel void gpu_set_outgrid(
    constant InterpArgs& args [[buffer(0)]],
    const device float*  in  [[buffer(1)]],   // [in_dim][in_dim][in_dim]
    device       float*  out [[buffer(2)]],   // [out_dim][out_dim][out_dim]
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]])
{
    uint xi = tid.x + bid.x*8;
    uint yi = tid.y + bid.y*8;
    uint zi = tid.z + bid.z*8;
    if (xi >= args.out_dim || yi >= args.out_dim || zi >= args.out_dim) return;

    // Cartesian coords of this output grid point
    float3 outpt = {
        xi * args.out_res + args.out_ox,
        yi * args.out_res + args.out_oy,
        zi * args.out_res + args.out_oz
    };

    // Apply inverse rotation: newpt = invQ * (outpt + untranslate - center) + center
    float x = outpt.x + args.utx;
    float y = outpt.y + args.uty;
    float z = outpt.z + args.utz;
    float3 newpt = quat_rotate(args.qa, args.qb, args.qc, args.qd,
                                x - 0, y - 0, z - 0); // rotation is around origin after untranslate
    newpt.x += args.cx;
    newpt.y += args.cy;
    newpt.z += args.cz;

    // Convert to fractional input grid coordinates
    float gx = (newpt.x - args.in_ox) / args.in_res;
    float gy = (newpt.y - args.in_oy) / args.in_res;
    float gz = (newpt.z - args.in_oz) / args.in_res;

    // Trilinear interpolation
    int xl = int(floor(gx)), xh = int(ceil(gx));
    int yl = int(floor(gy)), yh = int(ceil(gy));
    int zl = int(floor(gz)), zh = int(ceil(gz));

    uint idm = args.in_dim;

#define GET_PT(ix,iy,iz) ( \
    ((ix)<0||(ix)>=(int)idm||(iy)<0||(iy)>=(int)idm||(iz)<0||(iz)>=(int)idm) \
    ? 0.0f : in[(uint(ix)*idm + uint(iy))*idm + uint(iz)] )

    float p000 = GET_PT(xl,yl,zl), p001 = GET_PT(xl,yl,zh);
    float p010 = GET_PT(xl,yh,zl), p011 = GET_PT(xl,yh,zh);
    float p100 = GET_PT(xh,yl,zl), p101 = GET_PT(xh,yl,zh);
    float p110 = GET_PT(xh,yh,zl), p111 = GET_PT(xh,yh,zh);
#undef GET_PT

    float xd = (xh > xl) ? (gx - xl) / (xh - xl) : 0.0f;
    float yd = (yh > yl) ? (gy - yl) / (yh - yl) : 0.0f;
    float zd = (zh > zl) ? (gz - zl) / (zh - zl) : 0.0f;

    float c00 = p000*(1-xd) + p100*xd;
    float c01 = p001*(1-xd) + p101*xd;
    float c10 = p010*(1-xd) + p110*xd;
    float c11 = p011*(1-xd) + p111*xd;
    float c0  = c00*(1-yd)  + c10*yd;
    float c1  = c01*(1-yd)  + c11*yd;
    float c   = c0*(1-zd)   + c1*zd;

    out[(xi*args.out_dim + yi)*args.out_dim + zi] = c;
}
