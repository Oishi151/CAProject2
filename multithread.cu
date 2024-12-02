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
__global__ void mergeSort(int* array, int* temp, int size) 
{
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int width = 1; width < size; width *= 2) 
    {
        if (tid < size) 
        {
            int left = tid * 2 * width;
            int mid = min(left + width - 1, size - 1);
            int right = min(left + 2 * width - 1, size - 1);

            if (left <= right) 
            {
                int i = left;
                int j = mid + 1;
                int k = left;

                while (i <= mid && j <= right) 
                {
                    if (array[i] <= array[j]) 
                    {
                        temp[k++] = array[i++];
                    }
                     else 
                    {
                        temp[k++] = array[j++];
                    }
                }
                while (i <= mid) 
                {
                    temp[k++] = array[i++];
                }

                while (j <= right) 
                {
                    temp[k++] = array[j++];
                }

                for (i = left; i <= right; i++) 
                {
                    array[i] = temp[i];
                }
            }
        }
        __syncthreads();
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
    mergeSort<<<num_buckets, threads_per_block>>>(device_data, device_bucket_offsets, num_elements);

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
