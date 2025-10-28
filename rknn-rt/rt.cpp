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
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <queue>
#include <stack>
#include <vector>
#include <limits>
#include <random>
#include <utility>

#include <time.h>
#include <sys/time.h>

#include "rt.h"
#include "inf.h"

using namespace std;

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;


void printUsageAndExit( const char* argv0 )
{
    cerr << "Usage  : " << argv0 << " [options]\n";
    cerr << "Options: --infile | -if <filename>   Specify file for txt intput\n";
    cerr << "         --outfile | -of <filename>  Specify file for image output\n";
    cerr << "         --help | -h                 Print this usage message\n";
    exit( 1 );
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    cerr << "[" << setw( 2 ) << level << "][" << setw( 12 ) << tag << "]: "
              << message << "\n";
}

/* custom functions */
void print_reslt(uint32_t* rslt, Point* usr_list, uint32_t width, uint32_t height)
{
    // printf("RSLT: \n");
    uint32_t candidate_cnt = 0;
    for (uint32_t i = 0; i < height; i++) 
    {
        for (uint32_t j = 0; j < width; j++) 
        {
            if (rslt[i * width + j] == 1)
            {
                // printf("%d ", usr_list[ i * width + j ].id);
                candidate_cnt++;
            }
        }
        // printf("\n");
    }
    cout << "Found " << candidate_cnt << " RkNN results." << endl;
}

double get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time,NULL)){ 
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

/* end custom functions */


int main( int argc, char* argv[] )
{
    string infile_path;
    string outfile;
    string algorithm_name = "all";
    uint32_t k = 4;
    uint32_t q = 0;

    for( int i = 1; i < argc; ++i )
    {
        const string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--infile" || arg == "-if" )
        {
            if( i < argc - 1 )
            {
                infile_path = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if ( arg == "--algorithm" || arg == "-al" )
        {
            if( i < argc - 1 )
            {
                algorithm_name = argv[++i];
            }
        }
        else if ( arg == "-k" )
        {
            if (i < argc - 1 )
            {
                k = stoi(argv[++i]);
                if (k <= 0) 
                {
                    cerr << "Invalid value for k: " << k << ". It must be a positive integer." << endl;
                    exit(1);
                }
            }
            else
            {
                cerr << "Missing value for k." << endl;
                printUsageAndExit( argv[0] );
            }
        }
        else if ( arg == "-q" )
        {
            if (i < argc - 1 )
            {
                q = stoi(argv[++i]);
                if (q <= 0) 
                {
                    cerr << "Invalid value for q: " << q << ". It must be a positive number." << endl;
                    printUsageAndExit( argv[0] );
                }
            }
            else
            {
                cerr << "Missing value for q." << endl;
                printUsageAndExit( argv[0] );
            }
        }
        // else if( arg == "--outfile" || arg == "-of" )
        // {
        //     if( i < argc - 1 )
        //     {
        //         outfile = argv[++i];
        //     }
        //     else
        //     {
        //         printUsageAndExit( argv[0] );
        //     }
        // }
        else
        {
            cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        //
        // loading data
        //
        cout << "Loading data..." << endl;

        // open file
        ifstream infile(infile_path);
        if (!infile) 
        {
            cerr << "Error opening file: " << infile_path << endl;
            exit(1);
        }
        
        uint32_t fac_cnt;
        infile >> fac_cnt;

        Point *fac = new Point[fac_cnt];

        for (uint32_t i = 0; i < fac_cnt; ++i) 
        {
            infile >> fac[i].id;
            infile >> fac[i].x;
            infile >> fac[i].y;
        }

        uint32_t usr_cnt;
        infile >> usr_cnt;

        Point *usr = new Point[usr_cnt];

        for (uint32_t i = 0; i < usr_cnt; ++i) 
        {
            infile >> usr[i].id;
            infile >> usr[i].x;
            infile >> usr[i].y;
        }

        cout << "Loaded " << fac_cnt << " facilities and " << usr_cnt << " users." << endl;
        infile.close();

        // Determine space boundaries (bounding box of all facilities and users)
        double min_x = numeric_limits<double>::max();
        double max_x = numeric_limits<double>::lowest();
        double min_y = numeric_limits<double>::max();
        double max_y = numeric_limits<double>::lowest();

        for (uint32_t i = 0; i < fac_cnt; ++i) {
            min_x = min(min_x, fac[i].x);
            max_x = max(max_x, fac[i].x);
            min_y = min(min_y, fac[i].y);
            max_y = max(max_y, fac[i].y);
        }
        for (uint32_t i = 0; i < usr_cnt; ++i) {
            min_x = min(min_x, usr[i].x);
            max_x = max(max_x, usr[i].x);
            min_y = min(min_y, usr[i].y);
            max_y = max(max_y, usr[i].y);
        }
        Rectangle space_boundaries(min_x, min_y, max_x, max_y);

        int length = max_x - min_x + 2;
        int height = max_y - min_y + 2;
        cout << "Scene length: " << length << ", Scene height: " << height << endl << endl;

        double start_time, end_time;

        // Build R*-Tree
        cout << "building R*-trees..." << endl;
        start_time = get_wall_time();
        RStarTree fac_rtree;
        RStarTree usr_rtree;
        for (uint32_t i = 0; i < fac_cnt; ++i) 
        {
            fac_rtree.insert(fac[i]);
        }
        end_time = get_wall_time();
        cout << "R*-tree is built in " << end_time - start_time << "[s]." <<  endl << endl;

        //
        // scene construction
        //
        cout << "Constructing scene..." << endl;
        start_time = get_wall_time();
        Point query_point(fac[q].x, fac[q].y, fac[q].id);
        vector<Line> bisectors;

        // Initialize vertex list with the four corners of space boundaries
        vector<Vertex> vertices;
        vertices.push_back(Vertex(Point(min_x, min_y), true));  // bottom-left
        vertices.push_back(Vertex(Point(max_x, min_y), true));  // bottom-right
        vertices.push_back(Vertex(Point(min_x, max_y), true));  // top-left
        vertices.push_back(Vertex(Point(max_x, max_y), true));  // top-right

        // Priority queue-based bisector creation
        // Use min-heap priority queue (closest entries first)
        priority_queue<PQEntry, vector<PQEntry>, greater<PQEntry>> pq;

        // Initialize with root node
        auto root = fac_rtree.get_root();
        if (root) {
            PQEntry root_entry;
            root_entry.node = root;
            root_entry.distance = min_distance_to_rect(query_point, root->get_mbr());
            root_entry.is_point = false;
            pq.push(root_entry);
        }

        // Process entries from priority queue
        while (!pq.empty()) {
            PQEntry current = pq.top();
            pq.pop();

            if (current.is_point) {
                // This is a point - create a bisector if it's not the query point itself
                if (current.point.id != query_point.id) {
                    Line new_bisector = perpendicular_bisector(query_point, current.point);

                    // Update vertex pruning counts for existing vertices
                    for (auto& vertex : vertices) {
                        if (!new_bisector.is_on_valid_side(vertex.point)) {
                            vertex.pruning_count++;
                        }
                    }

                    // Find intersections with space boundaries
                    vector<Point> boundary_intersections = find_line_rectangle_intersections(
                        new_bisector, space_boundaries);

                    // Add new intersection vertices
                    for (const auto& intersection : boundary_intersections) {
                        // Check if this intersection point already exists (within tolerance)
                        bool exists = false;
                        for (const auto& vertex : vertices) {
                            if (abs(vertex.point.x - intersection.x) < 1e-9 &&
                                abs(vertex.point.y - intersection.y) < 1e-9) {
                                exists = true;
                                break;
                            }
                        }

                        if (!exists) {
                            Vertex new_vertex(intersection, true);
                            // Count how many existing bisectors can prune this new vertex
                            for (const auto& bisector : bisectors) {
                                if (!bisector.is_on_valid_side(intersection)) {
                                    new_vertex.pruning_count++;
                                }
                            }
                            vertices.push_back(new_vertex);
                        }
                    }

                    // Find intersections with existing bisectors
                    for (const auto& existing_bisector : bisectors) {
                        Point intersection;
                        if (find_line_intersection(new_bisector, existing_bisector, intersection)) {
                            // Check if intersection is within space boundaries
                            if (intersection.x >= space_boundaries.min_x &&
                                intersection.x <= space_boundaries.max_x &&
                                intersection.y >= space_boundaries.min_y &&
                                intersection.y <= space_boundaries.max_y) {

                                // Check if this intersection point already exists
                                bool exists = false;
                                for (const auto& vertex : vertices) {
                                    if (abs(vertex.point.x - intersection.x) < 1e-9 &&
                                        abs(vertex.point.y - intersection.y) < 1e-9) {
                                        exists = true;
                                        break;
                                    }
                                }

                                if (!exists) {
                                    Vertex new_vertex(intersection, false);
                                    // Count how many bisectors can prune this new vertex
                                    for (const auto& bisector : bisectors) {
                                        if (!bisector.is_on_valid_side(intersection)) {
                                            new_vertex.pruning_count++;
                                        }
                                    }
                                    vertices.push_back(new_vertex);
                                }
                            }
                        }
                    }

                    bisectors.push_back(new_bisector);

                    // Remove vertices with pruning_count >= k (they cannot be part of the RkNN region)
                    vertices.erase(
                        remove_if(vertices.begin(), vertices.end(),
                            [k](const Vertex& v) {
                                return v.pruning_count >= (int)k;
                            }),
                        vertices.end()
                    );
                }
            } else {
                // This is a node - check if it can be pruned by existing bisectors
                bool can_prune = true;

                // Calculate minimum distance from current node to vertices with pruning_count = k-1
                // or extreme boundary points with pruning_count < k
                Rectangle node_mbr = current.node->get_mbr();

                // First, identify extreme points on each boundary edge
                // For each boundary edge, we need the two points with minimum and maximum coordinate along that edge
                Point* left_min = nullptr;   // Leftmost point with min y on left boundary
                Point* left_max = nullptr;   // Leftmost point with max y on left boundary
                Point* right_min = nullptr;  // Rightmost point with min y on right boundary
                Point* right_max = nullptr;  // Rightmost point with max y on right boundary
                Point* bottom_min = nullptr; // Bottom point with min x on bottom boundary
                Point* bottom_max = nullptr; // Bottom point with max x on bottom boundary
                Point* top_min = nullptr;    // Top point with min x on top boundary
                Point* top_max = nullptr;    // Top point with max x on top boundary

                const double eps = 1e-9;  // Tolerance for boundary checking

                // Find extreme points among boundary vertices with pruning_count < k
                for (auto& vertex : vertices) {
                    if (vertex.is_boundary && vertex.pruning_count < (int)k) {
                        // Check which boundary this vertex is on
                        if (abs(vertex.point.x - space_boundaries.min_x) < eps) {
                            // Left boundary
                            if (!left_min || vertex.point.y < left_min->y) {
                                left_min = &vertex.point;
                            }
                            if (!left_max || vertex.point.y > left_max->y) {
                                left_max = &vertex.point;
                            }
                        }
                        if (abs(vertex.point.x - space_boundaries.max_x) < eps) {
                            // Right boundary
                            if (!right_min || vertex.point.y < right_min->y) {
                                right_min = &vertex.point;
                            }
                            if (!right_max || vertex.point.y > right_max->y) {
                                right_max = &vertex.point;
                            }
                        }
                        if (abs(vertex.point.y - space_boundaries.min_y) < eps) {
                            // Bottom boundary
                            if (!bottom_min || vertex.point.x < bottom_min->x) {
                                bottom_min = &vertex.point;
                            }
                            if (!bottom_max || vertex.point.x > bottom_max->x) {
                                bottom_max = &vertex.point;
                            }
                        }
                        if (abs(vertex.point.y - space_boundaries.max_y) < eps) {
                            // Top boundary
                            if (!top_min || vertex.point.x < top_min->x) {
                                top_min = &vertex.point;
                            }
                            if (!top_max || vertex.point.x > top_max->x) {
                                top_max = &vertex.point;
                            }
                        }
                    }
                }

                for (const auto& vertex : vertices) {
                    bool should_consider = false;

                    // Condition 1: vertex with pruning_count == k-1
                    if (vertex.pruning_count == (int)k - 1) {
                        should_consider = true;
                    }
                    // Condition 2: extreme boundary points with pruning_count < k
                    else if (vertex.is_boundary && vertex.pruning_count < (int)k) {
                        // Check if this vertex is one of the extreme points
                        if ((left_min && abs(vertex.point.x - left_min->x) < eps && abs(vertex.point.y - left_min->y) < eps) ||
                            (left_max && abs(vertex.point.x - left_max->x) < eps && abs(vertex.point.y - left_max->y) < eps) ||
                            (right_min && abs(vertex.point.x - right_min->x) < eps && abs(vertex.point.y - right_min->y) < eps) ||
                            (right_max && abs(vertex.point.x - right_max->x) < eps && abs(vertex.point.y - right_max->y) < eps) ||
                            (bottom_min && abs(vertex.point.x - bottom_min->x) < eps && abs(vertex.point.y - bottom_min->y) < eps) ||
                            (bottom_max && abs(vertex.point.x - bottom_max->x) < eps && abs(vertex.point.y - bottom_max->y) < eps) ||
                            (top_min && abs(vertex.point.x - top_min->x) < eps && abs(vertex.point.y - top_min->y) < eps) ||
                            (top_max && abs(vertex.point.x - top_max->x) < eps && abs(vertex.point.y - top_max->y) < eps)) {
                            should_consider = true;
                        }
                    }

                    if (should_consider &&
                        min_distance_to_rect(query_point, node_mbr) < query_point.distance_to(vertex.point)
                       ) {
                        can_prune = false;
                        break;
                    }
                }

                // If not pruned, add its children/points to the priority queue
                if (!can_prune) {
                    if (current.node->is_leaf) {
                        // Add points from leaf node
                        for (const auto& entry : current.node->entries) {
                            PQEntry point_entry;
                            point_entry.is_point = true;
                            point_entry.point = entry.point;
                            point_entry.distance = query_point.distance_to(entry.point);
                            point_entry.node = nullptr;
                            pq.push(point_entry);
                        }
                    } else {
                        // Add child nodes from internal node
                        for (const auto& entry : current.node->entries) {
                            if (entry.child) {
                                PQEntry child_entry;
                                child_entry.node = entry.child;
                                child_entry.distance = min_distance_to_rect(query_point, entry.child->get_mbr());
                                child_entry.is_point = false;
                                pq.push(child_entry);
                            }
                        }
                    }
                }
            }
        }
        // cout << "Total bisectors created: " << bisectors.size() << endl << endl;

        uint32_t n_meshes = bisectors.size();
        vector<TriangleMesh> meshes(n_meshes);
        uint32_t count_meshes = 0;
        uint32_t z = 1;

        for (const auto& bisector : bisectors) 
        {
            if (bisector.is_vertical) 
            {
                if (bisector.valid_side == 0) 
                {
                    // create right mesh (rectangle)
                    float2 right_lt = {(float)bisector.x_val, (float)height};
                    float2 right_rb = {(float)length, 0};
                    meshes[count_meshes++].addRectangle(right_lt, right_rb, z);
                }
                else 
                {
                    // create left mesh (rectangle)
                    float2 left_lt = {0, (float)height};
                    float2 left_rb = {(float)bisector.x_val, 0};
                    meshes[count_meshes++].addRectangle(left_lt, left_rb, z);
                }
            }
            else if (bisector.a == 0) 
            {
                if (bisector.valid_side == 0)
                {
                    // create upper mesh (rectangle)
                    float2 upper_lt = {0, (float)height};
                    float2 upper_rb = {(float)length, (float)bisector.b};
                    meshes[count_meshes++].addRectangle(upper_lt, upper_rb, z);
                }
                else 
                {
                    // create lower mesh (rectangle)
                    float2 lower_lt = {0, (float)bisector.b};
                    float2 lower_rb = {(float)length, 0};
                    meshes[count_meshes++].addRectangle(lower_lt, lower_rb, z);
                }
            }
            else
            {
                if ( bisector.a > 0 )
                {
                    if (bisector.valid_side == 0)
                    {
                        // create upper mesh (triangle)
                        float2 upper_m = {0, (float)height};
                        float2 upper_h = {((float)height - (float)bisector.b) / (float)bisector.a, (float)height};
                        float2 upper_l = {0, (float)bisector.b};
                        meshes[count_meshes++].addTriangle(upper_m, upper_h, upper_l, z);
                    }
                    else
                    {
                        // create lower mesh (triangle)
                        float2 lower_m = {(float)length, 0};
                        float2 lower_h = {(float)length, (float)length * (float)bisector.a + (float)bisector.b};
                        float2 lower_l = {(0 - (float)bisector.b) / (float)bisector.a, 0};
                        meshes[count_meshes++].addTriangle(lower_m, lower_h, lower_l, z);
                    }
                }
                else
                {
                    if (bisector.valid_side == 0)
                    {
                        // create upper mesh (triangle)
                        float2 upper_m = {(float)length, (float)height};
                        float2 upper_h = {((float)height - (float)bisector.b) / (float)bisector.a, (float)height};
                        float2 upper_l = {(float)length, (float)bisector.a * (float)length + (float)bisector.b};
                        meshes[count_meshes++].addTriangle(upper_m, upper_h, upper_l, z);
                    }
                    else
                    {
                        // create lower mesh (triangle)
                        float2 lower_m = {0, 0};
                        float2 lower_h = {0, (float)bisector.b};
                        float2 lower_l = {(0 - (float)bisector.b) / (float)bisector.a, 0};
                        meshes[count_meshes++].addTriangle(lower_m, lower_h, lower_l, z);
                    }
                }
            }
            z++;
        }
        
        end_time = get_wall_time();
        double scene_time = end_time - start_time;
        cout << "Depth of the scene: " << z << endl;
        cout << "Scene constructed in " << scene_time << "[s]." << endl << endl;

        vector<int2> ray_coords(usr_cnt);

        for (uint32_t i = 0; i < usr_cnt; ++i) 
        {
            ray_coords[i].x = usr[i].x;
            ray_coords[i].y = usr[i].y;
        }

        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK( cudaFree( 0 ) );
            int numDevices;
            CUDA_CHECK( cudaGetDeviceCount(&numDevices) );
            if (numDevices == 0)
                throw runtime_error("#optix: no CUDA capable devices found!");
            cout << "#optix: found " << numDevices << " CUDA devices" << endl;

            CUDA_CHECK( cudaSetDevice( 0 ) ); 
            
            // Initialize the OptiX API, loading all API entry points
            OPTIX_CHECK( optixInit() );
            cout << "#optix: successfully initialized optix... yay!"
                      << endl;

            // Specify context options
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = &context_log_cb;
            options.logCallbackLevel          = 0;

            // Associate a CUDA context (and therefore a specific GPU) with this
            // device context
            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }


        //
        // accel handling
        //
        OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;
        {
            cout << "Building OptiX acceleration structure..." << endl;
            start_time = get_wall_time();
            // Use default options for simplicity.  In a real use case we would want to
            // enable compaction, etc
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
            accel_options.motionOptions.numKeys = 0;

            vector<OptixBuildInput> triangle_inputs(meshes.size());
            vector<CUdeviceptr> md_vertices(meshes.size());
            vector<CUdeviceptr> md_indices(meshes.size());
            vector<uint32_t> triangle_input_flags(meshes.size());

            for (uint32_t meshID = 0; meshID < meshes.size(); meshID++) {
                // Malloc and Memcpy vertices on device
                // array<float3> &vertices = meshes[meshID].vertices.data();
                const size_t vertices_size = sizeof( float3 ) * meshes[meshID].vertices.size();
                CUdeviceptr d_vertices = 0;
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( d_vertices ),
                            meshes[meshID].vertices.data(),
                            vertices_size,
                            cudaMemcpyHostToDevice
                            ) );
                md_vertices[meshID] = d_vertices;

                // Malloc and Memcpy indicies on device
                // array<float3, 4> &vertices = meshes[meshID];
                const size_t indices_size = sizeof( float3 ) * meshes[meshID].indices.size();
                CUdeviceptr d_indices = 0;
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_indices ), indices_size ) );
                CUDA_CHECK( cudaMemcpy(
                            reinterpret_cast<void*>( d_indices ),
                            meshes[meshID].indices.data(),
                            indices_size,
                            cudaMemcpyHostToDevice
                            ) );
                md_indices[meshID] = d_indices;
                
                // Our build input is a simple list of non-indexed triangle vertices
                triangle_input_flags[meshID] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;

                triangle_inputs[meshID] = {};
                triangle_inputs[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                
                triangle_inputs[meshID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
                triangle_inputs[meshID].triangleArray.vertexStrideInBytes = sizeof(float3);
                triangle_inputs[meshID].triangleArray.numVertices         = static_cast<uint32_t>( meshes[meshID].vertices.size() );
                triangle_inputs[meshID].triangleArray.vertexBuffers       = &md_vertices[meshID];

                triangle_inputs[meshID].triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                triangle_inputs[meshID].triangleArray.indexStrideInBytes  = sizeof(int3);
                triangle_inputs[meshID].triangleArray.numIndexTriplets    = static_cast<uint32_t>( meshes[meshID].indices.size() );;
                triangle_inputs[meshID].triangleArray.indexBuffer         = md_indices[meshID];

                // in this example we have one SBT entry, and no per-primitive materials:   
                triangle_inputs[meshID].triangleArray.flags                       = &triangle_input_flags[meshID];
                triangle_inputs[meshID].triangleArray.numSbtRecords               = 1;
                triangle_inputs[meshID].triangleArray.sbtIndexOffsetBuffer        = 0; 
                triangle_inputs[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
                triangle_inputs[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0; 
            }
            
            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK( optixAccelComputeMemoryUsage(
                        context,
                        &accel_options,
                        triangle_inputs.data(),
                        (int)meshes.size(), // Number of build inputs
                        &gas_buffer_sizes
                        ) );
            CUdeviceptr d_temp_buffer_gas;
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &d_temp_buffer_gas ),
                        gas_buffer_sizes.tempSizeInBytes
                        ) );
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &d_gas_output_buffer ),
                        gas_buffer_sizes.outputSizeInBytes
                        ) );
            OPTIX_CHECK( optixAccelBuild(
                        context,
                        0,                  // CUDA stream
                        &accel_options,
                        triangle_inputs.data(),
                        (int)meshes.size(), // Number of build inputs
                        d_temp_buffer_gas,
                        gas_buffer_sizes.tempSizeInBytes,
                        d_gas_output_buffer,
                        gas_buffer_sizes.outputSizeInBytes,
                        &gas_handle,
                        nullptr,            // emitted property list
                        0                   // num emitted properties
                        ) );
            end_time = get_wall_time();
            
            // We can now free the scratch space buffer used during build and the vertex
            // inputs, since they are not needed by our trivial shading method
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
            for (int meshID = 0; meshID < (int)meshes.size(); meshID++) {
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( md_vertices[meshID] ) ) );
            }
        }
        double bvh_time = end_time - start_time;
        cout << "OptiX acceleration structure built in " << bvh_time << "[s]." << endl;

        //
        // Create module
        //
        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
            module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

            pipeline_compile_options.usesMotionBlur        = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues      = 3;
            pipeline_compile_options.numAttributeValues    = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

            size_t      inputSize  = 0;
            const char* input      = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "rt.cu", inputSize );

            OPTIX_CHECK_LOG( optixModuleCreate(
                        context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        input,
                        inputSize,
                        LOG, &LOG_SIZE,
                        &module
                        ) );
        }

        //
        // Create program groups
        //
        OptixProgramGroup raygen_prog_group   = nullptr;
        OptixProgramGroup miss_prog_group     = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &raygen_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        LOG, &LOG_SIZE,
                        &raygen_prog_group
                        ) );

            OptixProgramGroupDesc miss_prog_group_desc  = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &miss_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        LOG, &LOG_SIZE,
                        &miss_prog_group
                        ) );

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            hitgroup_prog_group_desc.hitgroup.moduleAH            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &hitgroup_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        LOG, &LOG_SIZE,
                        &hitgroup_prog_group
                        ) );
        }

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t    max_trace_depth  = 1;
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth            = max_trace_depth;
            OPTIX_CHECK_LOG( optixPipelineCreate(
                        context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups,
                        sizeof( program_groups ) / sizeof( program_groups[0] ),
                        LOG, &LOG_SIZE,
                        &pipeline
                        ) );

            OptixStackSizes stack_sizes = {};
            for( auto& prog_group : program_groups )
            {
                OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, pipeline ) );
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                     0,  // maxCCDepth
                                                     0,  // maxDCDEpth
                                                     &direct_callable_stack_size_from_traversal,
                                                     &direct_callable_stack_size_from_state, &continuation_stack_size ) );
            OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                    direct_callable_stack_size_from_state, continuation_stack_size,
                                                    1  // maxTraversableDepth
                                                    ) );
        }

        //
        // Set up shader binding table
        //
        start_time = get_wall_time();
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof( RayGenSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            RayGenSbtRecord rg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( raygen_record ),
                        &rg_sbt,
                        raygen_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof( MissSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord ms_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( miss_record ),
                        &ms_sbt,
                        miss_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            vector<HitGroupSbtRecord> hitgroup_records;
            for (uint32_t meshID = 0; meshID < meshes.size(); meshID++) {
                HitGroupSbtRecord hg_sbt;
                // hg_sbt.data.far_id = meshes[meshID].point_far;
                // hg_sbt.data.near_id = meshes[meshID].point_near;
                OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
                hitgroup_records.push_back(hg_sbt);
            }
            CUdeviceptr d_hitgroup_record;
            size_t hitgroup_record_size = sizeof(HitGroupSbtRecord) * (int)hitgroup_records.size();
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup_record ), hitgroup_record_size ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_hitgroup_record ),
                        hitgroup_records.data(),
                        hitgroup_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = d_hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
            sbt.hitgroupRecordCount         = (int)hitgroup_records.size();
        }
        end_time = get_wall_time();
        double stb_time = end_time - start_time;
        cout << "STB time: " << stb_time << "[s]." << endl << endl;

        start_time = get_wall_time();
        // Set result matrix
        sutil::CUDAOutputBuffer<uint32_t> result_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, usr_cnt, 1 );
        
        // Allocate device memory for array
        int2* d_ray_coords = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<int2**>(&d_ray_coords), usr_cnt * sizeof(int2)));
        CUDA_CHECK(cudaMemcpy(d_ray_coords, ray_coords.data(), usr_cnt * sizeof(int2), cudaMemcpyHostToDevice));
        end_time = get_wall_time();
        double ray_setup_time = end_time - start_time;
        cout << "setup ray to device time: " << ray_setup_time << "[s]." << endl << endl;

        //
        // launch
        //
        start_time = get_wall_time();
        {
            CUstream stream;
            CUDA_CHECK( cudaStreamCreate( &stream ) );

            Params params;
            params.rslt         = result_buffer.map();
            params.ray_coords   = d_ray_coords;
            params.k            = k;
            params.width        = usr_cnt;
            params.height       = 1;
            params.depth        = (float)z + 1.0f;
            params.handle       = gas_handle;

            CUdeviceptr d_param;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_param ),
                        &params, sizeof( params ),
                        cudaMemcpyHostToDevice
                        ) );


            OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, /*width=*/usr_cnt, /*height=*/1, /*depth=*/1 ) );
            CUDA_SYNC_CHECK();

            result_buffer.unmap();

            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );
        }
        end_time = get_wall_time();
        double ray_trace_time = end_time - start_time;
        cout << "OptiX launch time: " << ray_trace_time << "[s]." << endl << endl;

        //
        // Send results back to host
        //
        uint32_t* rslt;
        start_time = get_wall_time();
        {
            // // for image: output to file
            // sutil::ImageBuffer buffer;
            // buffer.data         = output_buffer.getHostPointer();
            // buffer.width        = width;
            // buffer.height       = height;
            // buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            // if( outfile.empty() )
            //     sutil::displayBufferWindow( argv[0], buffer );
            // else
            //     sutil::saveImage( outfile.c_str(), buffer, false );

            // for result: printout all result buffer
            rslt = result_buffer.getHostPointer();
        }
        end_time = get_wall_time();
        double copy_out_time = end_time - start_time;
        cout << "Send back to host time: " << copy_out_time << "[s]." << endl << endl;

        //
        // Print results
        //
        print_reslt(rslt, usr, usr_cnt, 1);
        cout << "Total time: " << scene_time + bvh_time + stb_time + ray_setup_time + ray_trace_time + copy_out_time << "[s]." << endl << endl;

        //
        // Cleanup
        //
        cout << "Cleaning up... " << endl;
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

            OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
            OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
            OPTIX_CHECK( optixModuleDestroy( module ) );

            OPTIX_CHECK( optixDeviceContextDestroy( context ) );
        }
        cout << "All done!" << endl;
    }
    catch( exception& e )
    {
        cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
