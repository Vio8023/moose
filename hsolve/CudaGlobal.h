#ifndef __CUDA_GLOBAL_H__
#define __CUDA_GLOBAL_H__

#define USE_CUDA

#ifdef USE_CUDA

#ifndef DEBUG_
#define DEBUG_
#endif

#ifdef DEBUG_

//#define DEBUG_VERBOSE
//#define DEBUG_STEP

#endif //DEBUG_

#define CUDA_ERROR_CHECK

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


#define BLOCK_WIDTH 256

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
 
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}
 
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}

#endif //USE_CUDA

#endif //  __CUDA_GLOBAL_H__
