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

void HSolveActive::resetDevice()
{
	cudaSafeCall(cudaDeviceReset());
	cudaSafeCall(cudaSetDevice(0));
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaThreadSynchronize());
}
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
	double                          * v_array,
	LookupColumn                    * column_array,                      
	double                          * caTable,
	const unsigned                  ca_nColumns,
	int                             * istate_power_map,
	LookupRow                       * ca_row_array,
	double                          * istate,
	int                             * instant_map,
	const unsigned                  set_size,
	double                          dt,
	const unsigned                  instant_z,
	const double 					min,
	const double 					max,
	const double					dx
	)
{
	int tID = threadIdx.x + blockIdx.x * blockDim.x;
	if ((tID)>= set_size) return;
	
	double x = v_array[tID];
	
	if ( x < min )
		x = min;
	else if ( x > max )
		x = max;

	double div = ( x - min ) / dx;
	double fraction = div - ( unsigned int )div;
	int rowIndex = (( unsigned int ) div ) * v_nColumns;

	int powerIndex = istate_power_map[tID];
	
	double * iTable;
	unsigned inCol;
	
	// if it is Z power and caRow
	if (powerIndex > 0 && ca_row_array[powerIndex].rowIndex != -1){
		rowIndex = ca_row_array[powerIndex].rowIndex;
		fraction = ca_row_array[powerIndex].fraction;
		iTable = caTable;
		inCol = ca_nColumns;
	}
	else {
		iTable = vTable;
		inCol = v_nColumns;
	}
	
	double a,b,C1,C2;
	double *ap, *bp;
	
	ap = iTable + rowIndex + column_array[tID].column;
	
	bp = ap + inCol;
	
	a = *ap;
	b = *bp;
	C1 = a + ( b - a ) * fraction;
	
	a = *( ap + 1 );
	b = *( bp + 1 );
	C2 = a + ( b - a ) * fraction;
	
	if(instant_map[tID] & getPower(powerIndex, instant_z)) {
		istate[tID] = C1 / C2;
	}
	
	else{
		double temp = 1.0 + dt / 2.0 * C2;
		istate[tID] = ( istate[tID] * ( 2.0 - temp ) + dt * C1 ) / temp;
	}  
}

void HSolveActive::copy_data(std::vector<LookupColumn>& column,
							 LookupColumn ** 			column_dd,
							 int * 						is_inited)
{
	if(!*is_inited)
	{
		*is_inited = 1;
		int size = column.size();
		printf("column size is :%d.\n", size);
		size = size <= 0?0:size;
		if(size)
		{
			cudaSafeCall(cudaMalloc((void**)column_dd, size * sizeof(LookupColumn)));
			cudaSafeCall(cudaMemcpy(*column_dd,
									&(column.front()),
									size * sizeof(LookupColumn),
									cudaMemcpyHostToDevice));
		}
	}	
}
void HSolveActive::advanceChannel_gpu(
	vector<double>&				     v_ac,
	vector<LookupRow>&               caRow,
	LookupColumn 					* column,                                           
	LookupTable&                     vTable,
	LookupTable&                     caTable,                       
	double                          * istate,
	int                             * instant_map,
	int                             * state_power_map,
	double                          dt,
	int 							set_size
	)
{
	double * v_ac_d;
	LookupRow * caRow_array_d;
	int * istate_power_map_d;  
	double * istate_d;
	int * instant_map_d;

	int caSize = caRow.size();
	
	cudaEvent_t mem_start, mem_stop;
	float mem_elapsed;
	cudaEventCreate(&mem_start);
	cudaEventCreate(&mem_stop);

	cudaEventRecord(mem_start);

	cudaSafeCall(cudaMalloc((void **)&v_ac_d, 				v_ac.size() * sizeof(double)));   
	cudaSafeCall(cudaMalloc((void **)&caRow_array_d, 		caRow.size() * sizeof(LookupRow)));  
 
	cudaSafeCall(cudaMalloc((void **)&istate_d, 			set_size * sizeof(double)));        
	cudaSafeCall(cudaMalloc((void **)&istate_power_map_d, 	set_size * sizeof(int))); 
	cudaSafeCall(cudaMalloc((void **)&instant_map_d, 		set_size * sizeof(int)));      

	cudaSafeCall(cudaMemcpy(v_ac_d, &v_ac.front(), sizeof(double) * v_ac.size(), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(caRow_array_d, &caRow.front(), sizeof(LookupRow) * caRow.size(), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(istate_power_map_d, state_power_map, set_size*sizeof(int), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(instant_map_d, instant_map, set_size*sizeof(int), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(istate_d, istate, set_size*sizeof(double), cudaMemcpyHostToDevice));
	
	if(!vTable.is_set())
	{
		vTable.set_is_set(true);
		vTable.copy_table();

	}
	if(!caTable.is_set())
	{
		caTable.set_is_set(true);
		caTable.copy_table();	
	}
	cudaCheckError();
	cudaEventRecord(mem_stop);
	cudaEventSynchronize(mem_stop);
	cudaEventElapsedTime(&mem_elapsed, mem_start, mem_stop);

	printf("GPU memory transfer time: %fms.\n", mem_elapsed);

	dim3 gridSize(set_size/BLOCK_WIDTH + 1, 1, 1);
	dim3 blockSize(BLOCK_WIDTH,1,1); 

	if(set_size <= BLOCK_WIDTH)
	{
		gridSize.x = 1;
		blockSize.x = set_size; 
	}    

	advanceChannel_kernel<<<gridSize,blockSize>>>( 
		vTable.get_table_d(),
		vTable.get_num_of_columns(),
		v_ac_d,
		column,
		caTable.get_table_d(),
		caTable.get_num_of_columns(),
		istate_power_map_d,
		caRow_array_d,
		istate_d,
		instant_map_d,
		set_size,
		dt,
		HSolveActive::INSTANT_Z,
		vTable.get_min(),
		vTable.get_max(),
		vTable.get_dx()
	);

	cudaCheckError(); 

	cudaSafeCall(cudaMemcpy(istate, istate_d, set_size * sizeof(double), cudaMemcpyDeviceToHost));

	cudaSafeCall(cudaDeviceSynchronize());    
 
	cudaSafeCall(cudaFree(v_ac_d));
	cudaSafeCall(cudaFree(caRow_array_d));
	cudaSafeCall(cudaFree(istate_d));
	cudaSafeCall(cudaFree(istate_power_map_d));
	cudaSafeCall(cudaFree(instant_map_d));
	
}
#endif
