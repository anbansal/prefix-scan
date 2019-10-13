#include "gpuScan.cuh"
#include "cudaKernels.cuh"
#include <iostream>
#include <cmath>
#define checkCudaError(check) _checkCudaError(check, __func__,__LINE__,__FILE__)
#define SHARED_MEMORY_BANKS 32
#define THREADS_PER_BLOCK 512
#define ELEMENTS_PER_BLOCK (2*THREADS_PER_BLOCK)
void _checkCudaError(cudaError_t err, const char *caller,int line,const char *fileName) {
  if (err != cudaSuccess) {
    std::cout<< "Error in:" <<caller<<std::endl<<
    "Line no. "<<line<<std::endl<<"File Name: "<<fileName<<cudaGetErrorString(err)<<std::endl;
    exit(0);
  }
}
int powerTwo_indexCal(int n){

  int x = 1;
	while (x < n) {
		x *= 2;
	}
	return x;
}
void gpuData::setData(int *in,int length){
  N = length;
  h_in = &in[0];
  h_out = new int[N];
  halfWayLoad = 1;
  offBits = log2(SHARED_MEMORY_BANKS);
}

gpuData::~gpuData(){
  delete[] h_out;
}

void printArray(int* in,int N){
  int incr = 1;
  if(N>128) incr = 63;
  for (int i = 0;i < N;i += incr){
    std::cout<<"["<<i<<"] "<< in[i]<<" ";
  }
    std::cout<<std::endl;
}

void gpuData::exclusiveScan(){


  int vectorLen = N*sizeof(int);
  checkCudaError(cudaMalloc((void **)&d_in,vectorLen )); //SAFE create the memory on GPU
  checkCudaError(cudaMalloc((void **)&d_out, vectorLen));//SAFE create the memory on GPU
  checkCudaError(cudaMemcpy(d_in, h_in, vectorLen, cudaMemcpyHostToDevice));//SAFE copy the memory from host to GPU
  checkCudaError(cudaMemcpy(d_out,h_out, vectorLen, cudaMemcpyHostToDevice));//SAFE copy the memory from host to GPU
  // Create cudaRunTime Event to record the timer
	cudaEvent_t start, stop;
  checkCudaError(cudaEventCreate(&start)); //SAFE create the start timer
  checkCudaError(cudaEventCreate(&stop)); // SAFE create the stop timer
  checkCudaError(cudaEventRecord(start)); // SAFE record the start timer to do the GPU Computation

  if(N > ELEMENTS_PER_BLOCK){
    scanLargerBlockSize(d_in,d_out,N);
  }
  else {

    scanSmallerBlockSize(d_in,d_out,N);
  }

  checkCudaError(cudaEventRecord(stop)); // record the end timer to do the GPU Computation
	checkCudaError(cudaEventSynchronize(stop));
	checkCudaError(cudaEventElapsedTime(&runTime, start, stop)); //compute the total gpu time

  checkCudaError(cudaMemcpy(h_out, d_out, vectorLen, cudaMemcpyDeviceToHost)); //SAFE copy the memory form gpu to host

  checkCudaError(cudaFree(d_out));  //SAFE free the memory on GPU
  checkCudaError(cudaFree(d_in));   //SAFE free the memory on GPU
  checkCudaError(cudaEventDestroy(start)); //SAFE destroy the GPU Timer event
  checkCudaError(cudaEventDestroy(stop)); //SAFE destroy the GPU Timer event

}
//do the exclusive scan if the number of elements are less then ELEMENTS_PER_BLOCK
void gpuData::scanSmallerBlockSize(int *d_input,int *d_output,int len){
  int blocks = 1;
  int threads = (len+1)/2;
  //***************************************MOST IMPORTANT STEP***************************************
  int powerTwo_index = powerTwo_indexCal(len); //without it the program will either not run or might give wrong result
  //*************************************************************************************************
  size_t sharedMemSize =  2*powerTwo_index*sizeof(int);
  exclusiveScanGPU<<< blocks,threads,sharedMemSize>>>(d_input, d_output, len,offBits,halfWayLoad,powerTwo_index);

}
//do the scan on GPU when the number of elements in the array are more than ELEMENTS_PER_BLOCK
void gpuData::scanLargerBlockSize(int *d_input,int *d_output,int len){
  //divide the array elements into the perfect mutiple of ELEMENTS_PER_BLOCK and remainder
  int remainder = len % ELEMENTS_PER_BLOCK;
  if (remainder == 0) {
    //if the number of elemet are a perfect mupltiple of ELEMENTS_PER_BLOCK
    scanPerfectBlockSize(d_input, d_output, len);
  }
  else {
    //else the number of elemet are not a perfect mupltiple of ELEMENTS_PER_BLOCK, then do it in two steps and add the data later on
    int lengthMultiple = len - remainder;
    scanPerfectBlockSize(d_input, d_output, lengthMultiple);
    int *startOfOutputArray = &(d_output[lengthMultiple]);
    scanSmallerBlockSize(&(d_input[lengthMultiple]),&(d_output[lengthMultiple]),  remainder);
    appendData<<<1, remainder>>>(&(d_output[lengthMultiple]), remainder, &(d_input[lengthMultiple - 1]), &(d_output[lengthMultiple - 1]));
  }
}

//do the scan on GPU when the number of elements in the array are muliple of ELEMENTS_PER_BLOCK
void gpuData::scanPerfectBlockSize(int *d_input,int *d_output,int len){
  int blocks = len/ELEMENTS_PER_BLOCK;
  int threads = THREADS_PER_BLOCK;
  size_t sharedMemSize =  2*(ELEMENTS_PER_BLOCK)*sizeof(int);

  //Stores the sums per block in a different array
  int *d_in_BlockSums, *d_out_BlockSums;
	checkCudaError(cudaMalloc((void **)&d_in_BlockSums, blocks * sizeof(int)));//SAFE create the memory on GPU
	checkCudaError(cudaMalloc((void **)&d_out_BlockSums, blocks * sizeof(int)));//SAFE fcreateree the memory on GPU

  exclusiveScanGPU<<< blocks,threads,sharedMemSize>>>(d_input, d_output, ELEMENTS_PER_BLOCK,offBits,halfWayLoad,d_in_BlockSums);

  //now do the exclusive scan on the blocks' sums
  const int numThreads = (blocks + 1) / 2;
	if (numThreads > THREADS_PER_BLOCK) {

		scanLargerBlockSize(d_in_BlockSums, d_out_BlockSums, blocks);
	}
	else {
		scanSmallerBlockSize(d_in_BlockSums, d_out_BlockSums, blocks);
	}

  //now add the blocks sums in the output array
	appendData<<<blocks, ELEMENTS_PER_BLOCK>>>(d_output, ELEMENTS_PER_BLOCK, d_out_BlockSums);

	checkCudaError(cudaFree(d_in_BlockSums));//SAFE free the memory on GPU
	checkCudaError(cudaFree(d_out_BlockSums));//SAFE free the memory on GPU
}
