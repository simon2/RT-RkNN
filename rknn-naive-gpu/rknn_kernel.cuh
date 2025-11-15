#ifndef RKNN_KERNEL_CUH
#define RKNN_KERNEL_CUH

#include <cuda_runtime.h>
#include <vector>

// Structure for GPU Point representation
struct GPUPoint {
    int x;
    int y;
    int id;
};

// Structure to hold distance and point index pairs
struct DistancePair {
    float distance;
    int point_idx;
};

// CUDA kernel function declarations
__global__ void computeDistancesKernel(
    const GPUPoint* users,
    const GPUPoint* facilities,
    float* distances,
    int num_users,
    int num_facilities
);

__global__ void findKNearestKernel(
    const float* distances,
    const GPUPoint* facilities,
    int* knn_results,
    int num_users,
    int num_facilities,
    int k,
    int query_facility_id
);

// Host-side wrapper function for GPU reverse k-NN computation
void gpuReverseKNN(
    const std::vector<GPUPoint>& h_users,
    const std::vector<GPUPoint>& h_facilities,
    int query_facility_id,
    int k,
    std::vector<int>& result_user_ids
);

// Utility function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#endif // RKNN_KERNEL_CUH