#include "rknn_kernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cstdio>
#include <algorithm>

// CUDA kernel to compute distances between all users and facilities
__global__ void computeDistancesKernel(
    const GPUPoint* users,
    const GPUPoint* facilities,
    float* distances,
    int num_users,
    int num_facilities)
{
    // Each thread handles one user-facility pair
    int user_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int facility_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (user_idx < num_users && facility_idx < num_facilities) {
        // Calculate Euclidean distance
        float dx = users[user_idx].x - facilities[facility_idx].x;
        float dy = users[user_idx].y - facilities[facility_idx].y;
        float dist = sqrtf(dx * dx + dy * dy);

        // Store distance in row-major order
        distances[user_idx * num_facilities + facility_idx] = dist;
    }
}

// Device function to perform insertion sort for k smallest elements
__device__ void insertionSortK(DistancePair* arr, int n, int k) {
    for (int i = 1; i < n && i < k; i++) {
        DistancePair key = arr[i];
        int j = i - 1;

        while (j >= 0 && arr[j].distance > key.distance) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// CUDA kernel to find k nearest facilities for each user and check if query facility is among them
__global__ void findKNearestKernel(
    const float* distances,
    const GPUPoint* facilities,
    int* knn_results,  // Output: 1 if user is in RkNN of query, 0 otherwise
    int num_users,
    int num_facilities,
    int k,
    int query_facility_id)
{
    int user_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (user_idx >= num_users) return;

    // Allocate shared memory for storing k nearest distances
    extern __shared__ DistancePair shared_distances[];
    DistancePair* user_distances = &shared_distances[threadIdx.x * k * 2];

    // Initialize with maximum distances
    for (int i = 0; i < k * 2 && i < num_facilities; i++) {
        user_distances[i].distance = FLT_MAX;
        user_distances[i].point_idx = -1;
    }

    // Find k smallest distances for this user
    int count = 0;
    for (int fac_idx = 0; fac_idx < num_facilities; fac_idx++) {
        float dist = distances[user_idx * num_facilities + fac_idx];

        if (count < k) {
            user_distances[count].distance = dist;
            user_distances[count].point_idx = fac_idx;
            count++;

            // Sort when we have k elements
            if (count == k) {
                insertionSortK(user_distances, count, k);
            }
        } else {
            // Check if this distance is smaller than the k-th smallest
            if (dist < user_distances[k-1].distance) {
                user_distances[k-1].distance = dist;
                user_distances[k-1].point_idx = fac_idx;

                // Re-sort to maintain order
                for (int i = k-1; i > 0 && user_distances[i].distance < user_distances[i-1].distance; i--) {
                    DistancePair temp = user_distances[i];
                    user_distances[i] = user_distances[i-1];
                    user_distances[i-1] = temp;
                }
            }
        }
    }

    // If we have fewer than k facilities, sort what we have
    if (count < k) {
        insertionSortK(user_distances, count, count);
    }

    // Check if query facility is among k nearest
    int is_rknn = 0;
    int check_limit = min(k, count);
    for (int i = 0; i < check_limit; i++) {
        if (user_distances[i].point_idx >= 0 &&
            facilities[user_distances[i].point_idx].id == query_facility_id) {
            is_rknn = 1;
            break;
        }
    }

    knn_results[user_idx] = is_rknn;
}

// Alternative kernel using a simpler approach without shared memory
__global__ void findKNearestSimpleKernel(
    const float* distances,
    const GPUPoint* facilities,
    int* knn_results,
    int num_users,
    int num_facilities,
    int k,
    int query_facility_id)
{
    int user_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (user_idx >= num_users) return;

    // Count how many facilities are closer than the query facility
    float query_distance = FLT_MAX;
    int query_fac_idx = -1;

    // Find the distance to the query facility
    for (int fac_idx = 0; fac_idx < num_facilities; fac_idx++) {
        if (facilities[fac_idx].id == query_facility_id) {
            query_distance = distances[user_idx * num_facilities + fac_idx];
            query_fac_idx = fac_idx;
            break;
        }
    }

    // If query facility not found, this user is not in RkNN
    if (query_fac_idx == -1) {
        knn_results[user_idx] = 0;
        return;
    }

    // Count facilities closer than the query facility
    int closer_count = 0;
    for (int fac_idx = 0; fac_idx < num_facilities; fac_idx++) {
        if (fac_idx != query_fac_idx) {
            float dist = distances[user_idx * num_facilities + fac_idx];
            if (dist < query_distance) {
                closer_count++;
                if (closer_count >= k) {
                    // Query facility is not in k-NN
                    knn_results[user_idx] = 0;
                    return;
                }
            }
        }
    }

    // Query facility is in k-NN
    knn_results[user_idx] = 1;
}

// Host-side wrapper function
void gpuReverseKNN(
    const std::vector<GPUPoint>& h_users,
    const std::vector<GPUPoint>& h_facilities,
    int query_facility_id,
    int k,
    std::vector<int>& result_user_ids)
{
    int num_users = h_users.size();
    int num_facilities = h_facilities.size();

    // Allocate device memory
    GPUPoint *d_users, *d_facilities;
    float *d_distances;
    int *d_knn_results;

    CUDA_CHECK(cudaMalloc(&d_users, num_users * sizeof(GPUPoint)));
    CUDA_CHECK(cudaMalloc(&d_facilities, num_facilities * sizeof(GPUPoint)));
    CUDA_CHECK(cudaMalloc(&d_distances, num_users * num_facilities * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_knn_results, num_users * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_users, h_users.data(), num_users * sizeof(GPUPoint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_facilities, h_facilities.data(), num_facilities * sizeof(GPUPoint), cudaMemcpyHostToDevice));

    // Initialize results to 0
    CUDA_CHECK(cudaMemset(d_knn_results, 0, num_users * sizeof(int)));

    // Compute distances using 2D grid
    dim3 distBlockSize(16, 16);
    dim3 distGridSize(
        (num_users + distBlockSize.x - 1) / distBlockSize.x,
        (num_facilities + distBlockSize.y - 1) / distBlockSize.y
    );

    computeDistancesKernel<<<distGridSize, distBlockSize>>>(
        d_users, d_facilities, d_distances, num_users, num_facilities
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Find k-nearest facilities for each user
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_users + threadsPerBlock - 1) / threadsPerBlock;

    // Use the simpler kernel for better stability
    findKNearestSimpleKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_distances, d_facilities, d_knn_results, num_users, num_facilities, k, query_facility_id
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<int> h_knn_results(num_users);
    CUDA_CHECK(cudaMemcpy(h_knn_results.data(), d_knn_results, num_users * sizeof(int), cudaMemcpyDeviceToHost));

    // Extract user IDs where result is 1
    result_user_ids.clear();
    for (int i = 0; i < num_users; i++) {
        if (h_knn_results[i] == 1) {
            result_user_ids.push_back(h_users[i].id);
        }
    }

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_users));
    CUDA_CHECK(cudaFree(d_facilities));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_knn_results));
}