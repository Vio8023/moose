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

__device__ __constant__ int instant_xyz_d[3];

void HSolveActive::resetDevice()
{
	cudaSafeCall(cudaDeviceReset());
	cudaSafeCall(cudaSetDevice(0));
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaThreadSynchronize());
}

void HSolveActive::copy_to_device(double ** v_row_array, double * v_row_temp, int size)
{
	cudaSafeCall(cudaMalloc((void**)v_row_array, sizeof(double) * size));
	cudaSafeCall(cudaMemcpy(*v_row_array, v_row_temp, sizeof(double) * size, cudaMemcpyHostToDevice));
}


// __device__ __host__ __inline__
// void print_binary(char * b, u64 data)
// {
// 	for (int i = 63; i >= 0; i--)
//     	b[63-i] = ((data >> i) & 1) == 1 ? '1' : '0';
//     b[64] = '\0';
// }

__global__
void advanceChannel_kernel(
	double                          * vTable,
	const unsigned                  v_nColumns,
	double							* v_row_array,
	LookupColumn                    * column_array,                      
	double                          * caTable,
	const unsigned                  ca_nColumns,
	ChannelData 					* channel,
	double                           * ca_row_array,
	double                          * istate,
	const unsigned                  channel_size,
	double                          dt,
	const unsigned					num_of_compartment
	)
{
	int tID = threadIdx.x + blockIdx.x * blockDim.x;
	int id = tID;
	if ((tID)>= channel_size) return;
	u64 data = channel[tID];
	if(get_compartment_index(data) >= num_of_compartment)
	{
		printf("id %d ch %d compartment index %d >= number of compartment %d.\n", 
			id, tID,
			get_compartment_index(data), num_of_compartment);

	}	
	if(get_compartment_index(data) >= num_of_compartment){
		printf("tID: %d is doing the following printing.\n", tID);
	}	
	
	if(get_compartment_index(data) >= num_of_compartment)
	{
		char b[65];
		print_binary(b, data);
		printf("data: %s\n", b);
	}	
	tID = get_state_index(data);
	if(get_compartment_index(data) >= num_of_compartment){
		printf("state index: %d\n", tID);
		printf("compartment index: %d\n", get_compartment_index(data));
		printf("Instant: %d\n", get_instant(data));
	}	
	double myrow = v_row_array[get_compartment_index(data)];
	// if(id == 0){
	// 	printf("myrow: %f\n", myrow);
	// }		
	double * iTable;
	unsigned inCol;
	
	bool xyz[3] = {get_x(data), get_y(data), get_z(data)};

	// if(id == 0){
	// 	printf("x: %d, y:%d, z:%d\n", xyz[0], xyz[1], xyz[2]);
	// }

	for(int i = 0; i < 3; ++i)
	{
		// if(id == 0){
		// 	printf("iteration: %d\n", i);
		// }	
		if(!xyz[i]) continue;
		// if it is Z power and caRow
		if (i == 2 && ca_row_array[get_ca_row_index(data)]!= -1.0f){
			myrow = ca_row_array[get_ca_row_index(data)];
			iTable = caTable;
			inCol = ca_nColumns;
			// if(id == 0){
			// 	printf("In branch, myrow: %f\n", myrow);
			// }				
		}
		else {
			iTable = vTable;
			inCol = v_nColumns;
		}
		
		double a,b,C1,C2;
		double *ap, *bp;
		// if(id == 0){
		// 	printf("column: %d\n", column_array[tID].column);
		// }	
		ap = iTable + int(myrow) + column_array[tID].column;
		
		bp = ap + inCol;
		
		a = *ap;
		// if(id == 0){
		// 	printf("[C1] a: %f\n", a);
		// }			
		b = *bp;
		// if(id == 0){
		// 	printf("[C1] b: %f\n", b);
		// }		
		C1 = a + ( b - a ) * (myrow - int(myrow));
		// if(id == 0){
		// 	printf("[C1] C1: %f\n", C1);
		// }		
		a = *( ap + 1 );
		// if(id == 0){
		// 	printf("[C2] a: %f\n", a);
		// }		
		b = *( bp + 1 );
		// if(id == 0){
		// 	printf("[C2] b: %f\n", b);
		// }		
		C2 = a + ( b - a ) * (myrow - int(myrow));
		// if(id == 0){
		// 	printf("[C2] C2: %f\n", C2);
		// }		

		// if(id == 0){
		// 	printf("Instant: %d, power instant: %d\n", get_instant(data), instant_xyz_d[i]);
		// }		

		if(get_instant(data) & instant_xyz_d[i]) {
			istate[tID + i] = C1 / C2;
			// if(id == 0){
			// 	printf("C1/C2 state: %f\n", istate[tID + i]);
			// }		
		}
		
		else{
			double temp = 1.0 + dt / 2.0 * C2;
			istate[tID] = ( istate[tID] * ( 2.0 - temp ) + dt * C1 ) / temp;
			// if(id == 0){
			// 	printf("branch state: %f\n", istate[tID + i]);
			// }				
		} 
		tID ++;
	} 
}

void HSolveActive::copy_data(std::vector<LookupColumn>& column,
							 LookupColumn ** 			column_dd,
							 int * 						is_inited,
							 vector<ChannelData>&		channel_data,
							 ChannelData ** 			channel_data_dd,
							 const int 					x,
							 const int 					y,
							 const int 					z)
{
	if(!(*is_inited))
	{
		*is_inited = 1;
		int size = column.size();
		printf("column size is :%d.\n", size);

		cudaSafeCall(cudaMalloc((void**)column_dd, size * sizeof(LookupColumn)));
		cudaSafeCall(cudaMemcpy(*column_dd,
								&(column.front()),
								size * sizeof(LookupColumn),
								cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMalloc((void**)channel_data_dd, channel_data.size() * sizeof(ChannelData)));
		cudaSafeCall(cudaMemcpy(*channel_data_dd,
								&(channel_data.front()),
								channel_data.size() * sizeof(ChannelData),
								cudaMemcpyHostToDevice));
		const int xyz[3] = {x,y,z};
		cudaSafeCall(cudaMemcpyToSymbol(instant_xyz_d, xyz, sizeof(int)*3, 0, cudaMemcpyHostToDevice));
	}	
}
void HSolveActive::advanceChannel_gpu(
	double *						     v_row_d,
	vector<double>&               	 caRow,
	LookupColumn 					* column,                                           
	LookupTable&                     vTable,
	LookupTable&                     caTable,                       
	double                          * istate,
	ChannelData 					* channel,
	double                          dt,
	int 							set_size,
	int 							channel_size,
	int 							num_of_compartment
	)
{
	double * caRow_array_d;
	double * istate_d;

	int caSize = caRow.size();
	
	cudaEvent_t mem_start, mem_stop;
	float mem_elapsed;
	cudaEventCreate(&mem_start);
	cudaEventCreate(&mem_stop);

	cudaEventRecord(mem_start);

	cudaSafeCall(cudaMalloc((void **)&caRow_array_d, 		caRow.size() * sizeof(double)));  
	cudaSafeCall(cudaMalloc((void **)&istate_d, 			set_size * sizeof(double)));   

	cudaSafeCall(cudaMemcpy(caRow_array_d, &caRow.front(), sizeof(double) * caRow.size(), cudaMemcpyHostToDevice));
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

	//printf("GPU memory transfer time: %fms.\n", mem_elapsed);

	dim3 gridSize(channel_size/BLOCK_WIDTH + 1, 1, 1);
	dim3 blockSize(BLOCK_WIDTH,1,1); 

	if(channel_size <= BLOCK_WIDTH)
	{
		gridSize.x = 1;
		blockSize.x = channel_size; 
	}    

	advanceChannel_kernel<<<gridSize,blockSize>>>( 
		vTable.get_table_d(),
		vTable.get_num_of_columns(),
		v_row_d,
		column,
		caTable.get_table_d(),
		caTable.get_num_of_columns(),
		channel,
		caRow_array_d,
		istate_d,
		channel_size,
		dt,
		num_of_compartment
	);

	cudaCheckError(); 

	cudaSafeCall(cudaMemcpy(istate, istate_d, set_size * sizeof(double), cudaMemcpyDeviceToHost));

	cudaSafeCall(cudaDeviceSynchronize());    
 
 	cudaSafeCall(cudaFree(v_row_d));
	cudaSafeCall(cudaFree(caRow_array_d));
	cudaSafeCall(cudaFree(istate_d));
}
#endif
