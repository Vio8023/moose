#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <vector>
#include <algorithm>
#include "CudaGlobal.h"
#include "RateLookup.h"
#include "HSolveActive.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <assert.h>
#include <thrust/system/system_error.h>
#include <thrust/copy.h>

#define USE_CUDA
#define DEBUG

#ifdef USE_CUDA

__inline__ __host__ __device__
unsigned int getPower(int power_map_val, const unsigned int Z)
{
	if(power_map_val < 0)
	{
		return (unsigned int)(- power_map_val);
	}
	else
	{
		return Z;
	}
}


__inline__ __host__ __device__
double get_c(double a, double b, LookupRow row)
{
	return a + ( b - a ) * row.fraction;
}

__global__
void advanceChannel_kernel(
                           double                          * vTable,
                           const unsigned                  v_nColumns,
                           LookupRow                       * v_row_array,
                           LookupColumn                    * column_array,                      
                           double                          * caTable,
                           const unsigned                  ca_nColumns,
                           int                             * istate_power_map,
                           LookupRow                       * ca_row_array,
                           double                          * istate,
                           int                             * instant_map,
                           const unsigned                  set_size,
                           double                          dt,
                           const unsigned                  instant_z
                           )
{
    int tID = threadIdx.x + blockIdx.x * blockDim.x;
    if ((tID)>= set_size) return;
    
    LookupRow row;
    
    double * iTable;
    unsigned inCol;
    
    // if it is Z power and caRow
    if (istate_power_map[tID] > 0 && ca_row_array[istate_power_map[tID]].rowIndex != -1){
        row = ca_row_array[istate_power_map[tID]];
        iTable = caTable;
        inCol = ca_nColumns;
    }
    else {
        row = v_row_array[tID];
        iTable = vTable;
        inCol = v_nColumns;
    }
    
    //double a, b, C1, C2;
    double *ap, *bp;
   
    ap = iTable + row.rowIndex + column_array[tID].column;
    
    bp = ap + inCol;
    
    //a = *ap;
    //printf("My tID is %d, rowIndex: %d, column: %d.\n", tID, row.rowIndex, column_array[tID].column);
   // printf("My tID is %d, C1apid: %d, C1bpid: %d.\n", tID, row.rowIndex + column_array[tID].column, inCol);
    //printf("My tID is %d, C1 a: %f, C1 b: %f.\n", tID, *ap, *bp);
    //b = *bp;
    //C1 = a + ( b - a ) * row.fraction;
    //printf("My tID is %d, C1 a: %f, C1 b: %f.\n", tID, a, b);
    //a = *( ap + 1 );
    //b = *( bp + 1 );
    //C2 = a + ( b - a ) * row.fraction;
    //printf("My tID is %d, C2 a: %f, C2 b: %f.\n", tID, a, b);
    //printf("My tID is %d, iPower : %d.\n", tID, iPower);
    //printf("My tID is %d, C1: %f, C2: %f.\n", tID, C1, C2);
    //printf("My tID is %d, my istate is %f. \n", tID, istate[tID]);
    //if(tID == 0) printf("set_size is %d\n", set_size);
    //assert(tID < set_size);
    
    if(instant_map[tID]&getPower(istate_power_map[tID], instant_z)) {
		//printf("tID: %d in iInstant&iPower\n", tID);
        istate[tID] = get_c(*ap, *bp, row) / get_c(*(ap+1), *(bp+1), row);
    }
    else{
		//printf("tID: %d in else branch\n", tID);
        //double temp = 1.0 + dt / 2.0 * get_c(*(ap+1), *(bp+1), row);
        istate[tID] = ( istate[tID] * ( 2.0 - (1.0 + dt / 2.0 * get_c(*(ap+1), *(bp+1), row)) ) + dt * get_c(*ap, *bp, row) ) / (1.0 + dt / 2.0 * get_c(*(ap+1), *(bp+1), row));
    }
    
    //printf("My tID is %d. Job finished!\n", tID);
    
}

__global__
void empty_kernel(double * state)
{
    int tID = threadIdx.x + blockIdx.x * blockDim.x;
    
    double my_state = state[tID];
    printf("[%d] state: %f.\n", tID, my_state);
    state[tID] = my_state + 1;
    state[tID] -= 1.0f;
    return;
}

void HSolveActive::advanceChannel_gpu(
                                            vector<LookupRow>&               vRow,
                                            vector<LookupRow>&               caRow,
                                            vector<LookupColumn>&            column,                                           
                                            LookupTable&                     vTable,
                                            LookupTable&                     caTable,                       
                                            double                          * istate,
                                            int                             * instant_map,
                                            int                             * state_power_map,
                                            double                          dt
)
{
    //cudaSafeCall(cudaDeviceReset());
    //cudaSafeCall(cudaDeviceSynchronize());
    LookupColumn * column_array_d;
    LookupRow * vRow_array_d;
    LookupRow * caRow_array_d;
    int * istate_power_map_d;  
    double * istate_d;
    int * instant_map_d;
    double * vTable_d;
    double * caTable_d;
    
    int set_size = column.size();
     
    int caSize = caRow.size();
    int vTableSize = vTable.get_num_of_points();
    int caTableSize = caTable.get_num_of_points();
    
    //printf("Start declaring memory...\n");
    cudaSafeCall(cudaMalloc((void **)&column_array_d, set_size*sizeof(LookupColumn)));
    cudaSafeCall(cudaMalloc((void **)&vRow_array_d, vRow.size()*sizeof(LookupRow)));   
    cudaSafeCall(cudaMalloc((void **)&caRow_array_d, caSize*sizeof(LookupRow)));    
   // cudaSafeCall(cudaMalloc((void **)&vTable_d, vTable.get_table().size()*sizeof(double)));        
    cudaSafeCall(cudaMalloc((void **)&caTable_d,caTable.get_table().size()*sizeof(double)));       
    //cudaSafeCall(cudaMalloc((void **)&istate_d, sizeof(double) * set_size));        
    cudaSafeCall(cudaMalloc((void **)&istate_power_map_d, sizeof(int) * set_size)); 
    cudaSafeCall(cudaMalloc((void **)&instant_map_d, sizeof(int) * set_size));      
            
    cudaSafeCall(cudaMemcpy(column_array_d, &column.front(), sizeof(LookupColumn) * column.size(), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(vRow_array_d, &vRow.front(), sizeof(LookupRow) * vRow.size(), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(caRow_array_d, &caRow.front(), sizeof(LookupRow) * caRow.size(), cudaMemcpyHostToDevice));
    
    thrust::device_vector<double> vTable_dv(vTable.get_table().begin(),
                                          vTable.get_table().end());
    vTable_d = thrust::raw_pointer_cast(vTable_dv.data());
    
    //cudaSafeCall(cudaMemcpy(vTable_d,&(vTable.get_table().front()),vTable.get_table().size()*sizeof(double),cudaMemcpyHostToDevice));
    //printf("caTable num of points: %d, caTable_ size: %d.\n", caTableSize, caTable.get_table().size());
    //printf("vTable num of points: %d, vTable_ size: %d.\n", vTableSize, vTable.get_table().size());
    //getchar();
    
    
    cudaSafeCall(cudaMemcpy(caTable_d,&(caTable.get_table().front()),caTable.get_table().size()*sizeof(double),cudaMemcpyHostToDevice));
    //cudaSafeCall(cudaMemcpy(istate_d, istate, set_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(istate_power_map_d, state_power_map, set_size*sizeof(int), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(instant_map_d, instant_map, set_size*sizeof(int), cudaMemcpyHostToDevice));
    
    thrust::device_vector<double> istate_dv(istate, istate+set_size);
    istate_d = thrust::raw_pointer_cast(istate_dv.data());
    //cudaSafeCall(cudaMemcpy(istate_d, istate, set_size*sizeof(double), cudaMemcpyHostToDevice));
    
    
    cudaCheckError();

   dim3 gridSize(set_size/BLOCK_WIDTH + 1, 1, 1);
   dim3 blockSize(BLOCK_WIDTH,1,1); 
   
   if(set_size <= BLOCK_WIDTH)
   {
        gridSize.x = 1;
        blockSize.x = set_size; 
   }    

    
    
    
    //getchar();
        //empty_kernel<<<gridSize,blockSize>>>(istate_d);
        //cudaDeviceSynchronize();
        //printf("Start advanceChannel kernel with %d threads...\n", set_size);
        //getchar();
        advanceChannel_kernel<<<gridSize,blockSize>>>( vTable_d,
                                                   vTable.get_num_of_columns(),
                                                   vRow_array_d,
                                                   column_array_d,
                                                   caTable_d,
                                                   caTable.get_num_of_columns(),
                                                   istate_power_map_d,
                                                   caRow_array_d,
                                                   istate_d,
                                                   instant_map_d,
                                                   set_size,
                                                   dt,
                                                   HSolveActive::INSTANT_Z
                                                   );
        cudaCheckError();                                                   
        if(cudaSuccess == cudaGetLastError())
        {
        std::vector<double> h_istate(set_size);
        thrust::copy(istate_dv.begin(), istate_dv.end(), h_istate.begin());
        
        std::copy(h_istate.begin(), h_istate.end(), istate);
        cudaSafeCall(cudaDeviceSynchronize());    
        } else {
          printf("Error in kernel launch (1)... Discard current round!\n");
          getchar();    
        }    
        cudaSafeCall(cudaFree(column_array_d));
        cudaSafeCall(cudaFree(vRow_array_d));
        cudaSafeCall(cudaFree(caRow_array_d));
        //cudaSafeCall(cudaFree(vTable_d));
        cudaSafeCall(cudaFree(caTable_d));
        //cudaSafeCall(cudaFree(istate_d));
        cudaSafeCall(cudaFree(istate_power_map_d));
        cudaSafeCall(cudaFree(instant_map_d));
    

    
   // printf("Exiting advanceCHannel kernel.\n");
    //getchar();
}
#endif
