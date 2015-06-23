#ifndef __CUDA_GLOBAL_H__
#define __CUDA_GLOBAL_H__

#define USE_CUDA

#ifdef USE_CUDA

#ifndef DEBUG_
#define DEBUG_
#endif

#ifdef DEBUG_

#define DEBUG_VERBOSE
#define DEBUG_STEP

#endif //DEBUG_

#define CUDA_ERROR_CHECK

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

typedef unsigned int u32;
typedef unsigned long long u64;
typedef u64 ChannelData;

const u64 X_SHIFT_BIT =63,
          Y_SHIFT_BIT =62,
          Z_SHIFT_BIT =61,
          CA_ROW_SHIFT_BIT =45,
          INSTANT_SHIFT_BIT =42,
          COMPARTMENT_SHIFT_BIT = 26,
          STATE_SHIFT_BIT = 0;
const u64 X_MASK = 1ull << X_SHIFT_BIT,
          Y_MASK = 1ull << Y_SHIFT_BIT,
          Z_MASK = 1ull << Z_SHIFT_BIT,
          CA_ROW_MASK = u64(0x10) << CA_ROW_SHIFT_BIT,
          INSTANT_MASK = u64(0x7) << INSTANT_SHIFT_BIT,
          COMPARTMENT_MASK = u64(0x3F) << COMPARTMENT_SHIFT_BIT,
          STATE_MASK = u64(0x3FFFFFF) << STATE_SHIFT_BIT;
inline
void pack_x(u64& data, int x)
{
    data |= (((u64)x << X_SHIFT_BIT) & X_MASK);
}

inline
void pack_y(u64& data, int y)
{
    data |= (((u64)y << Y_SHIFT_BIT) & Y_MASK);
}

inline
void pack_z(u64& data, int z)
{
    data |= (((u64)z << Z_SHIFT_BIT) & Z_MASK);
}

inline
void pack_ca_row_index(u64& data, int ca_row_index)
{
    data |= (((u64)ca_row_index << CA_ROW_SHIFT_BIT) & CA_ROW_MASK);
}

inline
void pack_instant(u64& data, int instant)
{
    data |= (((u64)instant << INSTANT_SHIFT_BIT) & INSTANT_MASK);
}

inline
void pack_compartment_index(u64& data, int compartment_index)
{
    data |= (((u64)compartment_index << COMPARTMENT_SHIFT_BIT) & COMPARTMENT_MASK);
}

inline
void pack_state_index(u64& data, int state_index)
{
    data |= (((u64)state_index << STATE_SHIFT_BIT) & STATE_MASK);
}

#define BLOCK_WIDTH 512

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
