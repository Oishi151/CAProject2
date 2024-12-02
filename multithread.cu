#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <vector>
#include <algorithm>

#define BLOCK_SIZE 256

// CUDA error checking macro
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// Kernel for counting the occurrences of each bit
__global__ void countKernel(int* input, int* counts, int n, int bit) {
    __shared__ int localCounts[2 * BLOCK_SIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    localCounts[threadIdx.x] = 0;
    localCounts[threadIdx.x + BLOCK_SIZE] = 0;

    if (tid < n) {
        int value = input[tid];
        int bin = (value >> bit) & 1;
        atomicAdd(&localCounts[bin * BLOCK_SIZE + threadIdx.x], 1);
    }
    __syncthreads();

    // Sum counts into global memory
    if (threadIdx.x < 2) {
        int sum = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += localCounts[i + threadIdx.x * BLOCK_SIZE];
        }
        atomicAdd(&counts[threadIdx.x], sum);
    }
}

// Kernel for prefix sum (scan) computation
__global__ void scanKernel(int* counts, int* offsets, int n) {
    __shared__ int temp[2];
    int tid = threadIdx.x;

    if (tid < 2) {
        temp[tid] = counts[tid];
    }
    __syncthreads();

    if (tid == 1) {
        offsets[1] = temp[0];
    }
    if (tid == 0) {
        offsets[0] = 0;
    }
    __syncthreads();
}

// Kernel for scattering elements into sorted positions
__global__ void scatterKernel(int* input, int* output, int* offsets, int n, int bit) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        int value = input[tid];
        int bin = (value >> bit) & 1;

        int index = atomicAdd(&offsets[bin], 1);
        output[index] = value;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [number of random integers to generate] [seed value for random number generation]\n";
        return 1;
    }

    int num_elements = std::atoi(argv[1]);
    int seed = std::atoi(argv[2]);

    // Generate random numbers on the host
    std::vector<int> host_input(num_elements);
    srand(seed);
    for (int i = 0; i < num_elements; ++i) {
        host_input[i] = rand();
    }

    // Allocate device memory
    int *device_input, *device_output, *device_counts, *device_offsets;
    CUDA_CHECK(cudaMalloc(&device_input, num_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_output, num_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_counts, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_offsets, 2 * sizeof(int)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice));

    // Define block and grid sizes
    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // CUDA timer
    cudaEvent_t start, stop;
    float elapsed_time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Perform radix sort (32-bit integers, 32 iterations)
    for (int bit = 0; bit < 32; ++bit) {
        // Reset counts
        CUDA_CHECK(cudaMemset(device_counts, 0, 2 * sizeof(int)));

        // Count occurrences of each bit
        countKernel<<<blocks_per_grid, threads_per_block>>>(device_input, device_counts, num_elements, bit);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute offsets
        scanKernel<<<1, 2>>>(device_counts, device_offsets, 2);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Scatter elements into sorted positions
        scatterKernel<<<blocks_per_grid, threads_per_block>>>(device_input, device_output, device_offsets, num_elements, bit);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap input and output
        std::swap(device_input, device_output);
    }

    // Stop timer
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    // Copy sorted data back to host
    CUDA_CHECK(cudaMemcpy(host_input.data(), device_input, num_elements * sizeof(int), cudaMemcpyDeviceToHost));

    std::cerr << "Elapsed time: " << elapsed_time / 1000.0f << " seconds" << std::endl;

    // Validate sorting
    if (std::is_sorted(host_input.begin(), host_input.end())) {
        std::cout << "Array sorted successfully." << std::endl;
    } else {
        std::cerr << "Sorting failed!" << std::endl;
    }

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_counts);
    cudaFree(device_offsets);

    return 0;
}
