#ifndef TESTSCANH
#define TESTSCANH
#include "utils.h"
#include <iomanip>
class testScan{
public:
  void testCPUGPU(Assignment *input){
    int num = input->getNum();
    std::vector<int> v = input->getInData();
    std::vector<int> vout_c = input->getOutData();
    int* temp = input->get_Ddata();
    std::vector<int> vout_g(temp,temp+num);
    std::vector<int> v1 (v.size());
    exclusive_scan(v.begin(),v.end(),v1.begin(),0);
    int endPoint = sumExclusive(v.begin(),v.end(),0);
    std::cout<<std::fixed<<std::setprecision(6);
    std::cout<<num<<"\t\t";
    bool cpuPass = false;
    bool gpuPass = false;
    bool result = true;
    if(!vout_c.empty()) result = (endPoint == vout_c.at(vout_c.size()-1));
    if ((v1 == vout_c) && result ) {
      cpuPass = true;
      std::cout<<"PASS\t\t";
      std::cout<<input->getRunTime()<<"\t\t";
    }
    else  {
        std::cout<<"FAIL\t\t";
        std::cout<<"NA\t\t";
    }

    if(!vout_g.empty()) result = (endPoint == vout_g.at(vout_g.size()-1));
    if ((v1 == vout_g) && result ) {
      gpuPass = true;
      std::cout<<"PASS\t\t";
      std::cout<<input->getGPURunTime()<<"\t\t";
    }
    else  {
      std::cout<<"FAIL\t\t";
      std::cout<<"NA\t\t";
    }
    if(cpuPass && gpuPass) std::cout<<(input->getRunTime()/input->getGPURunTime())<<std::endl;
    else std::cout<<"NA"<<std::endl;
  }
};
#endif
