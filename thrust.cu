#include <iostream>
#include <sstream>
#include <cstdlib>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

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

// Kernel function (not needed for Thrust sort, included for consistency)
__global__ void matavgKernel( ... )
{
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
        ss1 >> seed;
    }

    int sortPrint;
    {
        std::stringstream ss1( argv[3] );
        ss1 >> sortPrint;
    }

    if( sortPrint == 1 )
        printSorted = true;

    // get the random numbers
    array = makeRandArray( size, seed );

    /***********************************
     *
     create a cuda timer to time execution
     **********************************/
    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord( startTotal, 0 );
    /***********************************
     *
     end of cuda timer creation
     **********************************/

    /////////////////////////////////////////////////////////////////////
    /////////////////////// YOUR CODE HERE
    ///////////////////////
    /////////////////////////////////////////////////////////////////////
    /*
     *
     *
     *
     *
     *
     *
     *
     *
     You need to implement your kernel as a function at the top of this file.
     Here you must
     1) allocate device memory
     2) set up the grid and block sizes
     3) call your kenrnel
     4) get the result back from the GPU
     * to use the error checking code, wrap any cudamalloc functions as follows:
     *
     *
     *
     CudaSafeCall( cudaMalloc( &pointer_to_a_device_pointer, length_of_array * sizeof( int ) ) );
     * Also, place the following function call immediately after you call your kernel
     * ( or after any other cuda call that you think might be causing an error )
     CudaCheckError();
     */
    thrust::device_vector<int> device_data(array, array + size);
    thrust::sort(device_data.begin(), device_data.end());
    thrust::copy(device_data.begin(), device_data.end(), array);

    /***********************************
     *
     Stop and destroy the cuda timer
     **********************************/
    cudaEventRecord( stopTotal, 0 );
    cudaEventSynchronize( stopTotal );
    cudaEventElapsedTime( &timeTotal, startTotal, stopTotal );
    cudaEventDestroy( startTotal );
    cudaEventDestroy( stopTotal );
    /***********************************
     *
     end of cuda timer destruction
     **********************************/
    std::cerr << "Total time in seconds: " << timeTotal / 1000.0 << std::endl;
    if( printSorted ){
        ///////////////////////////////////////////////
        /// Your code to print the sorted array here //
        ///////////////////////////////////////////////
        for (int i = 0; i < size; ++i) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    }

    delete[] array;
    return 0;
}
