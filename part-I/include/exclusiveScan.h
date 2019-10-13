#ifndef EXCLUSIVESCANH
#define EXCLUSIVESCANH
#include <iostream>
#include <iterator>
#include <vector>
using namespace std;
class Assignment{
  int N;
  std::vector<int> in;
  std::vector<int> out;
  long runTime;

public:

  Assignment();

  Assignment(int num);

  Assignment(int num,int* inNum);

  Assignment(std::vector<int> inNum);

  void fillNum();

  void setNum(int num);
  int getNum(){ return N; }

  void setData(std::vector<int> inNum);
  std::vector<int> getInData(){ return in; }

  std::vector<int> getOutData(){ return out; }

  float getRunTime(){return runTime/1000000.0;}

  void run();
};

#endif
