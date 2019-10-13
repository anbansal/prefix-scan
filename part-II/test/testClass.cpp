#include "exclusiveScan.h"
#include "testScan.h"
#include <algorithm>

std::vector<int> setUpTests(int num){

  std::vector<int> N;
  srand (time(NULL));
  int bits = 1;
  int length = 1;
  while(N.size() < num && bits < 25){
    int temp = length << bits++;
    N.push_back(temp);
  }
  length = N.size();
  while (length < num){
    int temp = int((1<<bits)*((rand()%(1000-1 + 1) + 1)/10000.0));
    N.push_back(temp);
    length++;
  }
  std::sort(N.begin(),N.end());
  return N;
}

void doTest(int N){
  Assignment assignment(N);
  testScan runTest;
  assignment.run();
  assignment.runGPU();
  runTest.testCPUGPU(&assignment);
  cudaDeviceReset();
}
int main(){
/*
  int arr[] = {3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4};
  int N = sizeof(arr)/sizeof(arr[0]);
  std::vector<int> data (arr, arr + N );
*/
  std::cout<<std::endl<<"=========================================================================================================="<<std::endl;
  std::cout<<"Elements\tHost Test\tHost Time(ms.)\t\tGPU Test\tGPU Time(ms)\t\tSpeed Up"<<std::endl<<std::endl;
  std::vector<int> N;
  N = setUpTests(100);
  for (std::vector<int>::iterator it = N.begin();it < N.end();it++){
    doTest(*it);
  }
  return 0;
}
