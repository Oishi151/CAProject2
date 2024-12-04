#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std;

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

// CUDA error handling functions
inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __cudaCheckError(const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

// CUDA kernel for single-threaded merge sort
__global__ void mergeSort(int *array, int *temp, int size)
{
    // Ensure only one thread executes the sorting
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // Iterative merge sort
    for (int width = 1; width < size; width *= 2)
    {
        for (int i = 0; i < size; i += 2 * width)
        {
            int left = i;
            int mid = min(i + width - 1, size - 1);
            int right = min(i + 2 * width - 1, size - 1);

            // Merge the two halves
            int l = left, r = mid + 1, k = left;
            while (l <= mid && r <= right)
            {
                if (array[l] <= array[r])
                {
                    temp[k++] = array[l++];
                }
                else
                {
                    temp[k++] = array[r++];
                }
            }
            while (l <= mid)
            {
                temp[k++] = array[l++];
            }
            while (r <= right)
            {
                temp[k++] = array[r++];
            }

            // Copy back to the original array
            for (int j = left; j <= right; ++j)
            {
                array[j] = temp[j];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <size> <seed>" << endl;
        return -1;
    }

    int size = atoi(argv[1]);
    int seed = atoi(argv[2]);
    srand(seed);

    // Allocate and initialize the host array
    int *array = new int[size];
    for (int i = 0; i < size; ++i)
    {
        array[i] = rand() % 10000; // Random values between 0 and 9999
    }

    int *d_array, *d_temp;

    // Allocate memory on the device
    CudaSafeCall(cudaMalloc((void **)&d_array, size * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_temp, size * sizeof(int)));

    // Copy the array to the device
    CudaSafeCall(cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice));

    // Timer setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0); // Start timing

    // Launch the kernel with a single thread
    mergeSort<<<1, 1>>>(d_array, d_temp, size);
    CudaCheckError();

    cudaEventRecord(stop, 0); // Stop timing
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy the sorted array back to the host
    CudaSafeCall(cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Display the sorted array
   
    // Print time taken
    cout << "Time taken for sorting: " << milliseconds / 1000.0 << " seconds" << endl;

    // Free allocated memory
    delete[] array;
    cudaFree(d_array);
    cudaFree(d_temp);

    return 0;
}
