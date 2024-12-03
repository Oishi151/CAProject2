#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
using namespace std;

// Macro definitions for CUDA error checking
#define CUDA_CHECK_ERROR
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

// Error handling for CUDA calls
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    #ifdef CUDA_CHECK_ERROR
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
    #endif
}

inline void __cudaCheckError(const char *file, const int line) {
    #ifdef CUDA_CHECK_ERROR
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
    #endif
}

// Function to generate random array of integers
int* makeRandArray(const int size, const int seed) {
    srand(seed);
    int *array = new int[size];
    for (int i = 0; i < size; ++i) {
        array[i] = rand() % 100000; // Random integers between 0 and 99,999
    }
    return array;
}

// CUDA Kernel for parallel merge sort
__global__ void mergeSort(int* array, int* temp, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure tid is within bounds
    if (tid >= size) return;

    // Iterative merge sort
    for (int width = 1; width < size; width *= 2) {
        int left = tid * 2 * width;
        int mid = min(left + width - 1, size - 1);
        int right = min(left + 2 * width - 1, size - 1);

        // Check for valid ranges
        if (left >= size) return;

        int i = left, j = mid + 1, k = left;

        // Merge the two halves
        while (i <= mid && j <= right) {
            if (array[i] <= array[j]) {
                temp[k++] = array[i++];
            } else {
                temp[k++] = array[j++];
            }
        }

        while (i <= mid) {
            temp[k++] = array[i++];
        }

        while (j <= right) {
            temp[k++] = array[j++];
        }

        for (i = left; i <= right; i++) {
            array[i] = temp[i];
        }

        __syncthreads();
    }
}

int main() {
    const int seed = 42; // Fixed seed for reproducibility

    const int step = 50000;
    const int maxSize = 300000;
    const int minSize = 50000;

    // Open the CSV file for appending
    ofstream csvFile("SingleThread.csv", ios::app);
    if (!csvFile.is_open()) {
        cerr << "Error opening CSV file for writing!" << endl;
        return -1;
    }

    // Write the header only if the file is empty (first time run)
    if (csvFile.tellp() == 0) {
        csvFile << "Array Size,Time (seconds)" << endl;
    }

    for (int size = minSize; size <= maxSize; size += step) {
        cout << "Running merge sort for array size: " << size << endl;

        int *array = makeRandArray(size, seed); // Generate random array

        int *d_array, *d_temp;

        // Allocate memory on GPU
        CudaSafeCall(cudaMalloc((void**)&d_array, size * sizeof(int)));
        CudaSafeCall(cudaMalloc((void**)&d_temp, size * sizeof(int)));

        // Copy data to GPU
        CudaSafeCall(cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaDeviceSynchronize());
        // Timer setup
        cudaEvent_t startTotal, stopTotal;
        float timeTotal;
        cudaEventCreate(&startTotal);
        cudaEventCreate(&stopTotal);
        cudaEventRecord(startTotal, 0);

        // Launch mergeSort kernel
        mergeSort<<<1, 1>>>(d_array, d_temp, size);
cudaDeviceSynchronize();
        CudaCheckError();

        // Timer stop
        cudaEventRecord(stopTotal, 0);
        cudaEventSynchronize(stopTotal);
        cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
        cudaEventDestroy(startTotal);
        cudaEventDestroy(stopTotal);

        cerr << "Total time in seconds for size " << size << ": " << timeTotal / 1000.0 << endl;

        // Append results to CSV
        csvFile << size << "," << timeTotal / 1000.0 << endl;

        // Copy sorted array back to host
        CudaSafeCall(cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost));
        // Free allocated memory
        delete[] array;
        cudaFree(d_array);
        cudaFree(d_temp);
    }

    // Close the CSV file
    csvFile.close();
    return 0;
} 