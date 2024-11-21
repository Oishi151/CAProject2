#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cuda.h>

using namespace std;

/**********************************************************
 * **********************************************************
 *
 error checking stufff
 ***********************************************************
 ***********************************************************/
// Enable this for error checking
#define CUDA_CHECK_ERROR
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
    #pragma warning( push )
    #pragma warning( disable: 4127 )
    do
    {
        if ( cudaSuccess != err )
        {
            fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
            exit(-1 );
        }
    } while ( 0 );
    #pragma warning( pop )
#endif // CUDA_CHECK_ERROR
    return;
}
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
    #pragma warning( push )
    #pragma warning( disable: 4127 )
    do
    {
        cudaError_t err = cudaGetLastError();
        if ( cudaSuccess != err )
        {
            fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n", file, line, cudaGetErrorString( err ) );
            exit(-1 );
        }
        err = cudaThreadSynchronize();
        if( cudaSuccess != err )
        {
            fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n", file, line, cudaGetErrorString( err ) );
            exit(-1 );
        }
    } while ( 0 );
    #pragma warning( pop )
#endif // CUDA_CHECK_ERROR
    return;
}
/***************************************************************
 * **************************************************************
 *
 end of error checking stuff
 ****************************************************************
 ***************************************************************/

// Function to generate random numbers
int * makeRandArray( const int size, const int seed ) {
    srand( seed );
    int * array = new int[ size ];
    for( int i = 0; i < size; i ++ ) {
        array[i] = std::rand() % 1000000;
    }
    return array;
}

// CUDA kernel for Bubble Sort
__global__ void bubble_sort(int *array, int num_elements) {
    for (int i = 0; i < num_elements - 1; ++i) {
        for (int j = 0; j < num_elements - 1 - i; ++j) {
            if (array[j] > array[j + 1]) {
                // Swap elements
                int temp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = temp;
            }
        }
    }
}

int main( int argc, char* argv[] )
{
    int * array;
    int size, seed;
    bool printSorted = false;

    if( argc < 4 ){
        std::cerr << "usage: " << argv[0] << " [amount of random nums to generate] [seed value for rand] [1 to print sorted array, 0 otherwise]" << std::endl;
        exit(-1 );
    }

    {
        std::stringstream ss1( argv[1] );
        ss1 >> size;
    }

    {
        std::stringstream ss1( argv[2] );
