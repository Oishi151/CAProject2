// multi_thread_sort.cu
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
__global__ void sort_buckets(int *buckets, int num_buckets, int max_val) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int bucket_size = max_val / num_buckets;

    for (int i = 0; i < num_buckets - 1; ++i) {
        for (int j = 0; j < num_buckets - 1 - i; ++j) {
            if (buckets[tid * bucket_size + j] > buckets[tid * bucket_size + j + 1]) {
                int temp = buckets[tid * bucket_size + j];
                buckets[tid * bucket_size + j] = buckets[tid * bucket_size + j + 1];
                buckets[tid * bucket_size + j + 1] = temp;
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

    // Sort using CUDA kernel
    int num_threads = 1024;
    int num_buckets = num_elements / num_threads;
    sort_buckets<<<num_threads, 1>>>(device_data, num_buckets, num_elements);

    // Transfer data back to host
    cudaMemcpy(host_data, device_data, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    // Print sorted numbers
    for (int i = 0; i < num_elements; ++i) {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;

    delete[] host_data;
    cudaFree(device_data);
    return 0;
}
