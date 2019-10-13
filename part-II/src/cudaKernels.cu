#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include <stdlib.h>
#include "cudaKernels.cuh"
#define BANKCONFLICTS_FREE_OFFSET(n,x) ((n) >> x)


__global__ void exclusiveScanGPU(int *inData, int *outData, int N,int offsetBits,int halfWayLoad,int powerTwo_index){
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

  //compute the indexes to fill the shared memory;each thread insert two values in the shared memory as per the algorithm

  //consecutve load
  int firstInd = 2 * threadID;
  int secondInd = 2 * threadID + 1;

  //if Halfway load is true then fill up the half way
  if(halfWayLoad == 1){
    firstInd = threadID;
  	secondInd = threadID + (N / 2);
  }
  //compute the offset to avoid the shared memory bank conflict
	int bankOffsetFirstInd = BANKCONFLICTS_FREE_OFFSET(firstInd,offsetBits);
	int bankOffsetSecondInd = BANKCONFLICTS_FREE_OFFSET(secondInd,offsetBits);

  //fill up the shared memory
	if (threadID < N) {
		temp[firstInd + bankOffsetFirstInd] = inData[firstInd];
		temp[secondInd + bankOffsetSecondInd] = inData[secondInd];
	}
	else {
    temp[firstInd + bankOffsetFirstInd] = 0;
		temp[secondInd + bankOffsetSecondInd] = 0;
	}
// Start the Reduction step for computing the sums
	int offset = 1;

	for (int d = powerTwo_index >> 1; d > 0; d >>= 1)
	{
		__syncthreads(); //synchronize all the threads after they have completed the previous reduction step
		if (threadID < d)
		{
			int firstInd = offset * (2 * threadID + 1) - 1;
			int secondInd = offset * (2 * threadID + 2) - 1;
      firstInd += BANKCONFLICTS_FREE_OFFSET(firstInd,offsetBits);
      secondInd += BANKCONFLICTS_FREE_OFFSET(secondInd,offsetBits);
			temp[secondInd] += temp[firstInd];
		}
		offset *= 2;
	}

	if (threadID == 0) {
		temp[powerTwo_index - 1 + BANKCONFLICTS_FREE_OFFSET(powerTwo_index - 1,offsetBits)] = 0; // Setting up the last element to zero
	}

	for (int d = 1; d < powerTwo_index; d *= 2) // Now Down-Sweep Phase and trace back the sums
	{
		offset >>= 1;
		__syncthreads(); //synchronize all the threads after they have completed the previous down-sweep step
		if (threadID < d)
		{
			int firstInd = offset * (2 * threadID + 1) - 1;
			int secondInd = offset * (2 * threadID + 2) - 1;
			firstInd += BANKCONFLICTS_FREE_OFFSET(firstInd,offsetBits);
			secondInd += BANKCONFLICTS_FREE_OFFSET(secondInd,offsetBits);

			int t = temp[firstInd];
			temp[firstInd] = temp[secondInd];
			temp[secondInd] += t;
		}
	}

	__syncthreads();//synchronize all the threads after they have completed the Balanced Trees step

//now fill up the outData array
	if (threadID < N) {
		outData[firstInd] = temp[firstInd + bankOffsetFirstInd];
		outData[secondInd] = temp[secondInd + bankOffsetSecondInd];
	}
}
__global__ void exclusiveScanGPU(int *inData, int *outData, int N,int offsetBits,int halfWayLoad,int *sums) {
  extern __shared__ int temp[];// allocated on invocation
  int threadID = threadIdx.x;
	int blockID = blockIdx.x;
	int blockOffset = blockID * N;

  //compute the indexes to fill the shared memory;each thread insert two values in the shared memory as per the algorithm

  //consecutve load
  int firstInd = 2 * threadID;
  int secondInd = 2 * threadID + 1;

  //if Halfway load is true then fill up the half way
  if(halfWayLoad == 1){
    firstInd = threadID;
    secondInd = threadID + (N / 2);
  }


  //compute the offset to avoid the shared memory bank conflict
	int bankOffsetFirstInd = BANKCONFLICTS_FREE_OFFSET(firstInd,offsetBits);
	int bankOffsetSecondInd = BANKCONFLICTS_FREE_OFFSET(secondInd,offsetBits);

  //fill up the shared memory
	temp[firstInd + bankOffsetFirstInd] = inData[blockOffset + firstInd];
	temp[secondInd + bankOffsetSecondInd] = inData[blockOffset + secondInd];


  // Start the Reduction step for computing the sums
  	int offset = 1;
  	for (int d = N >> 1; d > 0; d >>= 1)
  	{
  		__syncthreads(); //synchronize all the threads after they have completed the previous reduction step
  		if (threadID < d)
  		{
  			int firstInd = offset * (2 * threadID + 1) - 1;
  			int secondInd = offset * (2 * threadID + 2) - 1;
        firstInd += BANKCONFLICTS_FREE_OFFSET(firstInd,offsetBits);
        secondInd += BANKCONFLICTS_FREE_OFFSET(secondInd,offsetBits);
  			temp[secondInd] += temp[firstInd];
  		}
  		offset *= 2;
  	}
	__syncthreads(); //synchronize all the threads so that sum array can be filled up

	if (threadID == 0) {
		sums[blockID] = temp[N - 1 + BANKCONFLICTS_FREE_OFFSET(N - 1,offsetBits)];
		temp[N - 1 + BANKCONFLICTS_FREE_OFFSET(N - 1,offsetBits)] = 0; // Setting up the last element to zero
	}

  for (int d = 1; d < N; d *= 2) // Now Down-Sweep Phase and trace back the sums
	{
		offset >>= 1;
		__syncthreads(); //synchronize all the threads after they have completed the previous down-sweep step
		if (threadID < d)
		{
			int firstInd = offset * (2 * threadID + 1) - 1;
			int secondInd = offset * (2 * threadID + 2) - 1;
			firstInd += BANKCONFLICTS_FREE_OFFSET(firstInd,offsetBits);
			secondInd += BANKCONFLICTS_FREE_OFFSET(secondInd,offsetBits);

			int t = temp[firstInd];
			temp[firstInd] = temp[secondInd];
			temp[secondInd] += t;
		}
	}

	__syncthreads();//synchronize all the threads after they have completed the Balanced Trees step
	outData[blockOffset + firstInd] = temp[firstInd + bankOffsetFirstInd];
	outData[blockOffset + secondInd] = temp[secondInd + bankOffsetSecondInd];
}
__global__ void appendData(int *outData, int lengthToAdd, int *inDataAdd) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * lengthToAdd;
	outData[blockOffset + threadID] += inDataAdd[blockID];
}
__global__ void appendData(int *outData, int lengthToAdd, int *inDataAdd1, int *inDataAdd2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;

	int blockOffset = blockID * lengthToAdd;

	outData[blockOffset + threadID] +=  inDataAdd1[blockID] + inDataAdd2[blockID];
}
