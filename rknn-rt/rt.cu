//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "rt.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>


extern "C" {
__constant__ Params params;
}


// for this simple example, we have a single ray type
enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };


extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const int ix = idx.x;
    const int iy = idx.y;
    
    // Set ray to parallel ray projection
    float3 ray_origin, ray_direction;
    int2 ray_coord = params.ray_coords[ix];
    ray_origin = { (float)ray_coord.x, (float)ray_coord.y, 0.f };
    ray_direction = { 0.f, 0.f, 1.f };

    // Initialize payloads
    uint32_t rslt = 1;      // 1 means "the q is my knn"
    uint32_t cnt = 0;

    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,                // Min intersection distance
        params.depth,               // Max intersection distance
        0.0f,                // rayTime -- used for motion blur
        OptixVisibilityMask( 255 ), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE,    // SBT offset   -- See SBT discussion
        RAY_TYPE_COUNT,      // SBT stride   -- See SBT discussion
        SURFACE_RAY_TYPE,    // missSBTIndex -- See SBT discussion
        rslt,                // Payload 0
        cnt                  // Payload 1
    );

    // Record results in our output raster
    params.rslt[ix] = rslt;
}


extern "C" __global__ void __miss__ms()
{
    /* No miss program logic needed */
}


extern "C" __global__ void __closesthit__ch()
{
    /* No closesthit program logic needed */
}

extern "C" __global__ void __anyhit__ah()
{

    // setting cost
    uint32_t cnt = optixGetPayload_1();
    cnt++;

    if (cnt >= params.k) {
        optixSetPayload_0( 0 );
        optixTerminateRay();
    }
    else
    {
        optixSetPayload_1(cnt);
        optixIgnoreIntersection();
    }
}
