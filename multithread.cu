#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <vector>

using namespace std;

// Function to generate random numbers
void generate_random_numbers(int *array, int num_elements, int seed) {
    srand(seed);
    for (int i = 0; i < num_elements; ++i) {
        array[i] = rand();
    }
}

// CUDA kernel for sorting each bucket using Thrust
__global__ void sort_buckets(int *buckets, int *bucket_offsets, int num_buckets, int bucket_size) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // Determine the start and end index for the current bucket
    int start_idx = bucket_offsets[bid];
    int end_idx = (bid == num_buckets - 1) ? bucket_size : bucket_offsets[bid + 1];

    // Sort the bucket using Bubble Sort (You can replace this with Thrust sort for better performance)
    for (int i = start_idx; i < end_idx - 1; ++i) {
        for (int j = start_idx; j < end_idx - 1 - (i - start_idx); ++j) {
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
    int num_buckets = 1024;       // Number of buckets (You can adjust this value based on your requirement)
    int bucket_size = (num_elements + num_buckets - 1) / num_buckets;

    // Allocate memory for bucket offsets
    int *host_bucket_offsets = new int[num_buckets];
    int *device_bucket_offsets;
    cudaMalloc(&device_bucket_offsets, num_buckets * sizeof(int));

    // Initialize bucket offsets
    for (int i = 0; i < num_buckets; ++i) {
        host_bucket_offsets[i] = i * bucket_size;
    }

    // Transfer bucket offsets to device
    cudaMemcpy(device_bucket_offsets, host_bucket_offsets, num_buckets * sizeof(int), cudaMemcpyHostToDevice);

    //std::cout << "Number of threads per block: " << threads_per_block << std::endl;
    //std::cout << "Number of buckets: " << num_buckets << std::endl;
    std::cout << "Total number of threads: " << threads_per_block * num_buckets << std::endl;

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

    // Sort each bucket using CUDA kernel
    sort_buckets<<<num_buckets, threads_per_block>>>(device_data, device_bucket_offsets, num_buckets, num_elements);

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
    delete[] host_bucket_offsets;
    cudaFree(device_data);
    cudaFree(device_bucket_offsets);
    return 0;
}
