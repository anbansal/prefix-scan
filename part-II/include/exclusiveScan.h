#ifndef EXCLUSIVESCANH
#define EXCLUSIVESCANH
#include <iostream>
#include <iterator>
#include <vector>
#include "gpuScan.cuh"
using namespace std;
class Assignment{
  int N;
  std::vector<int> in;
  std::vector<int> out;
  long runTime;
  gpuData d_data;

public:

  Assignment();

  Assignment(int num);

  Assignment(int num,int* inNum);

  Assignment(std::vector<int> inNum);

  void init_Ddata();

  void fillNum();

  void setNum(int num);
  int getNum(){ return N; }

  void setData(std::vector<int> inNum);
  std::vector<int> getInData(){ return in; }

  std::vector<int> getOutData(){ return out; }

  float getRunTime(){return runTime/1000000.0;}
  float getGPURunTime(){return d_data.getRunTime();}
  void run();
  void runGPU();


  int* get_Ddata(){return d_data.getOutData();}

};

#endif
