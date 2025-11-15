#ifndef INF_KERNEL_H
#define INF_KERNEL_H

#include <vector>

// Structure to represent a point on the GPU
struct CudaPoint {
    double x, y;
    int id;
};

// Structure to represent a line on the GPU
struct CudaLine {
    double a, b;       // For y = ax + b (non-vertical lines)
    double x_val;      // For vertical lines: x = x_val
    bool is_vertical;
    int valid_side;    // 1 for above(when vertical: right), 0 for below(when vertical: left)
};

// Function to perform GPU-based RkNN verification
// This function will be called from inf.cpp
void gpu_get_rknn_candidates(
    const std::vector<CudaPoint>& users,           // User points to check
    const std::vector<CudaLine>& bisectors,        // Bisectors from filtering phase
    std::vector<CudaPoint>& rknn_candidates,       // Output: RkNN candidates
    int k                                           // k value for RkNN
);

// Initialize CUDA (call once at startup)
void cuda_init();

// Clean up CUDA resources (call once at shutdown)
void cuda_cleanup();

// Function to check if CUDA is available
bool is_cuda_available();

// Function to get CUDA device properties
void print_cuda_device_info();

#endif // INF_KERNEL_H