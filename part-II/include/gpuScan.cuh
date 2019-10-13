#ifndef GPUSCANCUH
#define GPUSCANCUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
class gpuData{

  int *h_in,*h_out,*d_in,*d_out;
  int N;
  float runTime;
  int offBits;
  int halfWayLoad;
public:
  void setData(int* in,int length);
  void setHalfWayLoad(int halfWayLoadNew){halfWayLoad = halfWayLoadNew;}
  void setoffBits(int offBitsNew){offBits = offBitsNew;}
  float getRunTime(){return runTime;}
  int* getOutData(){return h_out;}
  void scanPerfectBlockSize(int *inData,int *outData,int len);
  void scanSmallerBlockSize(int *inData,int *outData,int len);
  void scanLargerBlockSize(int *inData,int *outData,int len);
  void exclusiveScan();
  ~gpuData();
};

#endif
