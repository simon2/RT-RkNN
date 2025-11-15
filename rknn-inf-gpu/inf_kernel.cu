#include "inf_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cstdio>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Device function to check if a point is on the valid side of a line
__device__ bool is_on_valid_side_device(const CudaPoint& point, const CudaLine& line) {
    if (line.valid_side == 1) {
        // Valid side is above (for non-vertical) or right (for vertical)
        if (line.is_vertical) {
            return point.x >= line.x_val;  // Right side of vertical line
        } else {
            return point.y >= (line.a * point.x + line.b);  // Above or on the line
        }
    } else {
        // Valid side is below (for non-vertical) or left (for vertical)
        if (line.is_vertical) {
            return point.x <= line.x_val;  // Left side of vertical line
        } else {
            return point.y <= (line.a * point.x + line.b);  // Below or on the line
        }
    }
}


// CUDA kernel for checking users against bisectors
// Each thread processes one user point
__global__ void check_users_kernel(
    const CudaPoint* users,
    const CudaLine* bisectors,
    int* violations_count,  // Output: violation count for each user
    int num_users,
    int num_bisectors,
    int k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_users) return;

    // Count violations for this user
    int violations = 0;

    for (int i = 0; i < num_bisectors; i++) {
        if (!is_on_valid_side_device(users[tid], bisectors[i])) {
            violations++;
            if (violations >= k) {
                break;  // Early termination
            }
        }
    }

    violations_count[tid] = violations;
}


// Main function to perform GPU-based RkNN verification
void gpu_get_rknn_candidates(
    const std::vector<CudaPoint>& users,
    const std::vector<CudaLine>& bisectors,
    std::vector<CudaPoint>& rknn_candidates,
    int k
) {
    if (users.empty() || bisectors.empty()) {
        return;
    }

    int num_users = users.size();
    int num_bisectors = bisectors.size();

    // Allocate device memory
    CudaPoint* d_users;
    CudaLine* d_bisectors;
    int* d_violations_count;

    CUDA_CHECK(cudaMalloc(&d_users, num_users * sizeof(CudaPoint)));
    CUDA_CHECK(cudaMalloc(&d_bisectors, num_bisectors * sizeof(CudaLine)));
    CUDA_CHECK(cudaMalloc(&d_violations_count, num_users * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_users, users.data(), num_users * sizeof(CudaPoint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bisectors, bisectors.data(), num_bisectors * sizeof(CudaLine), cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_users + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    check_users_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_users, d_bisectors, d_violations_count, num_users, num_bisectors, k
    );

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<int> violations_count(num_users);
    CUDA_CHECK(cudaMemcpy(violations_count.data(), d_violations_count,
                          num_users * sizeof(int), cudaMemcpyDeviceToHost));

    // Process results - add users with violations < k to candidates
    rknn_candidates.clear();
    for (int i = 0; i < num_users; i++) {
        if (violations_count[i] < k) {
            rknn_candidates.push_back(users[i]);
        }
    }

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_users));
    CUDA_CHECK(cudaFree(d_bisectors));
    CUDA_CHECK(cudaFree(d_violations_count));
}

// Initialize CUDA
void cuda_init() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        exit(1);
    }

    // Set device 0 as default
    CUDA_CHECK(cudaSetDevice(0));

    // Warm up CUDA context
    CUDA_CHECK(cudaFree(0));
}

// Clean up CUDA resources
void cuda_cleanup() {
    CUDA_CHECK(cudaDeviceReset());
}

// Check if CUDA is available
bool is_cuda_available() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }

    return true;
}

// Print CUDA device information
void print_cuda_device_info() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions: [" << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max grid dimensions: [" << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    }
}