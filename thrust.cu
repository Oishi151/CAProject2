#include <iostream>
#include <sstream>
#include <cstdlib>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

using namespace std;

/**********************************************************
 * **********************************************************
 *
 error checking stuff
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

int main( int argc, char* argv[] )
{
    if( argc != 3 ){
        std::cerr << "usage: " << argv[0] << " [number of random integers to generate] [seed value for random number generation]" << std::endl;
        exit(-1 );
    }

    int size, seed;

    {
        std::stringstream ss1( argv[1] );
        ss1 >> size;
    }

    {
        std::stringstream ss1( argv[2] );
        ss1 >> seed;
    }

    // get the random numbers
    int *array = makeRandArray(size, seed);

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

    // Transfer data to device
    thrust::device_vector<int> device_data(array, array + size);

    // Sort using Thrust
    thrust::sort(device_data.begin(), device_data.end());

    // Transfer data back to host
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

    // Print the sorted array
    //for (int i = 0; i < size; ++i) {
        //std::cout << array[i] << " ";
    //}
    std::cout << std::endl;

    delete[] array;
    return 0;
}
