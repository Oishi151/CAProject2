#include <iostream>
#include <cstdlib>
#include <cuda.h>

// Function to generate random numbers
void generate_random_numbers(int *array, int num_elements, int seed) {
    srand(seed);
    for (int i = 0; i < num_elements; ++i) {
        array[i] = rand();
    }
}

// CUDA kernel for sorting within each bucket
__global__ void sort_buckets(int *buckets, int num_elements) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Simple Bubble Sort within each thread's portion
    for (int i = 0; i < num_elements - 1; ++i) {
        for (int j = 0; j < num_elements - 1 - i; ++j) {
            if (buckets[j] > buckets[j + 1]) {
                int temp = buckets[j];
                buckets[j] = buckets[j + 1];
                buckets[j + 1] = temp;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [number of random integers to generate] [seed value for random number generation]\n";
        return 1;
    }

    int num_elements = std::atoi(argv[1]);
    int seed = std::atoi(argv[2]);

    // Generate random numbers
    int *host_data = new int[num_elements];
    generate_random_numbers(host_data, num_elements, seed);

    // Allocate memory on device
    int *device_data;
    cudaMalloc(&device_data, num_elements * sizeof(int));

    // Transfer data to device
    cudaMemcpy(device_data, host_data, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    // Determine the number of threads and blocks based on the requirement
    int threads_per_block = 256;  // You can adjust this value based on your requirement
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    int total_threads = threads_per_block * blocks_per_grid;

    //std::cout << "Number of threads per block: " << threads_per_block << std::endl;
    //std::cout << "Number of blocks per grid: " << blocks_per_grid << std::endl;
    std::cout << "Total number of threads: " << total_threads << std::endl;

    /***********************************
     *
     create a cuda timer to time execution
     **********************************/
    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal, 0);
    /***********************************
     *
     end of cuda timer creation
     **********************************/

    // Sort using CUDA kernel
    sort_buckets<<<blocks_per_grid, threads_per_block>>>(device_data, num_elements);

    /***********************************
     *
     Stop and destroy the cuda timer
     **********************************/
    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);
    cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
    /***********************************
     *
     end of cuda timer destruction
     **********************************/
    std::cerr << "Total time in seconds: " << timeTotal / 1000.0 << std::endl;

    // Transfer data back to host
    cudaMemcpy(host_data, device_data, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    // Print sorted numbers (commented out for large arrays)
    // for (int i = 0; i < num_elements; ++i) {
    //     std::cout << host_data[i] << " ";
    // }
    std::cout << std::endl;

    delete[] host_data;
    cudaFree(device_data);
    return 0;
}

