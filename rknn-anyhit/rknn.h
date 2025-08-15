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

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// #include "customParam.h"
#include "point.h"

//
// structs and functions for OptiX
//
struct Params
{
    uint32_t*       rslt;
    int2*           ray_coords;
    uint32_t        k;
    uint32_t        q;    
    uint32_t        width;
    uint32_t        height;
    float           depth;
    OptixTraversableHandle handle;
};


struct RayGenData
{ /* No data needed */ };


struct MissData
{ /* No data needed */ };


struct HitGroupData
{
    uint32_t    far_id;
    uint32_t    near_id;
};

//
// structs and functions for generating meshes
//
struct TriangleMesh 
{    
    // add aligned square with center and size using two meshes
    void addTriangle(float2 a, float2 b, float2 c, uint32_t z);
    void addRectangle(float2 lt, float2 rb, uint32_t z);
    
    std::vector<float3> vertices;
    std::vector<int3>   indices;
    // float3              color;
    uint32_t            point_near;
    uint32_t            point_far;
};

void TriangleMesh::addTriangle(float2 m, float2 h, float2 l, uint32_t z)
{
    // set vertices
    float3 a, b, c;
    a.x = m.x;     a.y = m.y;     a.z = z;
    b.x = h.x;     b.y = h.y;     b.z = z;
    c.x = l.x;     c.y = l.y;     c.z = z;

    vertices.push_back( a );
    vertices.push_back( b );
    vertices.push_back( c );
    
    // set indices 
    indices.push_back({ 0, 1, 2 });
}

void TriangleMesh::addRectangle(float2 lt, float2 rb, uint32_t z)
{
    // set vertices
    float3 a, b, c, d;
    a.x = lt.x;     a.y = lt.y;     a.z = z;
    b.x = lt.x;     b.y = rb.y;     b.z = z;
    c.x = rb.x;     c.y = rb.y;     c.z = z;
    d.x = rb.x;     d.y = lt.y;     d.z = z;

    vertices.push_back( a );
    vertices.push_back( b );
    vertices.push_back( c );
    vertices.push_back( d );
    
    // set indices 
    indices.push_back({ 0, 1, 2 });
    indices.push_back({ 0, 2, 3 });

}

//
// structs and functions for validation and evaluation
//
void print_reslt(uint32_t* rslt, uint32_t usr_list[], uint32_t width, uint32_t height);

double get_wall_time();
