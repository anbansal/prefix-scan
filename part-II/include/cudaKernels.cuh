#ifndef CUDAKERNELSH
#define CUDAKERNELSH

__global__ void exclusiveScanGPU(int *inData, int *outData, int N,int offsetBits,int halfWayLoad,int powerTwo_index);
__global__ void exclusiveScanGPU(int *inData, int *outData, int N,int offsetBits,int halfWayLoad,int *sums);
__global__ void appendData(int *outData, int lengthToAdd, int *inDataAdd);
__global__ void appendData(int *outData, int lengthToAdd, int *inDataAdd1, int *inDataAdd2) ;

#endif
