#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#include "CudaGlobal.h"

#include "RateLookup.h"
#include "HSolveActive.h"

#ifdef USE_CUDA

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/system_error.h>
#include <thrust/copy.h>

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
	LookupColumn col = column_array[tID];
	
	int iInstant = instant_map[tID];
	int powerIndex = istate_power_map[tID];
	int iPower = getPower(powerIndex, instant_z);
	
	double * iTable;
	unsigned inCol;
	
	// if it is Z power and caRow
	if (powerIndex > 0 && ca_row_array[powerIndex].rowIndex != -1){
		row = ca_row_array[powerIndex];
		iTable = caTable;
		inCol = ca_nColumns;
	}
	else {
		row = v_row_array[tID];
		iTable = vTable;
		inCol = v_nColumns;
	}
	
	double a,b,C1,C2;
	double *ap, *bp;
	
	ap = iTable + row.rowIndex + col.column;
	
	bp = ap + inCol;
	
	a = *ap;
	b = *bp;
	C1 = a + ( b - a ) * row.fraction;
	
	a = *( ap + 1 );
	b = *( bp + 1 );
	C2 = a + ( b - a ) * row.fraction;
	
	if(iInstant&iPower) {
		istate[tID] = C1 / C2;
	}
	
	else{
		double temp = 1.0 + dt / 2.0 * C2;
		istate[tID] = ( istate[tID] * ( 2.0 - temp ) + dt * C1 ) / temp;
	}  
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
	
	cudaSafeCall(cudaMalloc((void **)&vRow_array_d, 		vRow.size() * sizeof(LookupRow)));   
	cudaSafeCall(cudaMalloc((void **)&caRow_array_d, 		caRow.size() * sizeof(LookupRow)));  

    cudaSafeCall(cudaMalloc((void **)&vTable_d, 			vTable.get_table().size() * sizeof(double)));        
	cudaSafeCall(cudaMalloc((void **)&caTable_d,			caTable.get_table().size()* sizeof(double)));    

	cudaSafeCall(cudaMalloc((void **)&column_array_d, 		set_size * sizeof(LookupColumn)));   
	cudaSafeCall(cudaMalloc((void **)&istate_d, 			set_size * sizeof(double)));        
	cudaSafeCall(cudaMalloc((void **)&istate_power_map_d, 	set_size * sizeof(int))); 
	cudaSafeCall(cudaMalloc((void **)&instant_map_d, 		set_size * sizeof(int)));      

	cudaSafeCall(cudaMemcpy(column_array_d, &column.front(), sizeof(LookupColumn) * column.size(), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(vRow_array_d, &vRow.front(), sizeof(LookupRow) * vRow.size(), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(caRow_array_d, &caRow.front(), sizeof(LookupRow) * caRow.size(), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(caTable_d,&(caTable.get_table().front()),caTable.get_table().size()*sizeof(double),cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(vTable_d, &(vTable.get_table().front()), vTable.get_table().size()*sizeof(double), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(istate_power_map_d, state_power_map, set_size*sizeof(int), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(instant_map_d, instant_map, set_size*sizeof(int), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(istate_d, istate, set_size*sizeof(double), cudaMemcpyHostToDevice));
		
	cudaCheckError();

	dim3 gridSize(set_size/BLOCK_WIDTH + 1, 1, 1);
	dim3 blockSize(BLOCK_WIDTH,1,1); 

	if(set_size <= BLOCK_WIDTH)
	{
		gridSize.x = 1;
		blockSize.x = set_size; 
	}    

	advanceChannel_kernel<<<gridSize,blockSize>>>( 
		vTable_d,
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

	cudaSafeCall(cudaMemcpy(istate, istate_d, set_size * sizeof(double), cudaMemcpyDeviceToHost));

	cudaSafeCall(cudaDeviceSynchronize());    
 
	cudaSafeCall(cudaFree(column_array_d));
	cudaSafeCall(cudaFree(vRow_array_d));
	cudaSafeCall(cudaFree(caRow_array_d));
	cudaSafeCall(cudaFree(vTable_d));
	cudaSafeCall(cudaFree(caTable_d));
	cudaSafeCall(cudaFree(istate_d));
	cudaSafeCall(cudaFree(istate_power_map_d));
	cudaSafeCall(cudaFree(instant_map_d));
	
}
#endif
