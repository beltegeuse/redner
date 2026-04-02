// OptiX 9.1 device programs for redner ray tracing.
// Compiled to PTX at build time and embedded in the host binary.

#include <optix.h>
#include "optix_launch_params.h"

extern "C" {
__constant__ LaunchParams params;
}

// ---------- Closest-hit ray generation ----------
extern "C" __global__ void __raygen__closest()
{
    const unsigned int idx = optixGetLaunchIndex().x;
    if (idx >= params.num_rays) return;

    const OptiXRayData &ray = params.rays[idx];

    unsigned int p0, p1, p2;  // payload: t, tri_id, inst_id

    optixTrace(
        params.traversable,
        make_float3(ray.org_x, ray.org_y, ray.org_z),
        make_float3(ray.dir_x, ray.dir_y, ray.dir_z),
        ray.tmin,
        ray.tmax,
        0.0f,                        // rayTime
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_NONE,
        0,   // SBT offset
        1,   // SBT stride
        0,   // miss SBT index
        p0, p1, p2);

    OptiXHitData &hit = params.hits[idx];
    hit.t      = __uint_as_float(p0);
    hit.tri_id = static_cast<int>(p1);
    hit.inst_id = static_cast<int>(p2);
}

// ---------- Occlusion (any-hit) ray generation ----------
extern "C" __global__ void __raygen__occlusion()
{
    const unsigned int idx = optixGetLaunchIndex().x;
    if (idx >= params.num_rays) return;

    const OptiXRayData &ray = params.rays[idx];

    unsigned int p0, p1, p2;

    optixTrace(
        params.traversable,
        make_float3(ray.org_x, ray.org_y, ray.org_z),
        make_float3(ray.dir_x, ray.dir_y, ray.dir_z),
        ray.tmin,
        ray.tmax,
        0.0f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        0,   // SBT offset
        1,   // SBT stride
        0,   // miss SBT index
        p0, p1, p2);

    OptiXHitData &hit = params.hits[idx];
    hit.t       = __uint_as_float(p0);
    hit.tri_id  = static_cast<int>(p1);
    hit.inst_id = static_cast<int>(p2);
}

// ---------- Closest hit ----------
extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(static_cast<unsigned int>(optixGetPrimitiveIndex()));
    optixSetPayload_2(static_cast<unsigned int>(optixGetInstanceId()));
}

// ---------- Miss ----------
extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(__float_as_uint(-1.0f));
    optixSetPayload_1(static_cast<unsigned int>(-1));
    optixSetPayload_2(static_cast<unsigned int>(-1));
}
