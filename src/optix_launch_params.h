#pragma once

#include <optix_types.h>

// Shared between host (scene.cpp) and device (optix_programs.cu).
// Keep this header free of host-only includes.

struct OptiXRayData {
    float org_x, org_y, org_z;
    float tmin;
    float dir_x, dir_y, dir_z;
    float tmax;
};

struct OptiXHitData {
    float t;
    int   tri_id;
    int   inst_id;
};

struct LaunchParams {
    OptixTraversableHandle traversable;
    OptiXRayData*  rays;
    OptiXHitData*  hits;
    unsigned int   num_rays;
};
