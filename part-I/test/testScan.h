#ifndef TESTSCANH
#define TESTSCANH
#include "utils.h"
#include <iomanip>
class testScan{
public:
  void test(Assignment* input){
    int num = input->getNum();
    std::vector<int> v = input->getInData();
    std::vector<int> vout = input->getOutData();
    std::vector<int> v1 (v.size());
    exclusive_scan(v.begin(),v.end(),v1.begin(),0);
    int endPoint = sumExclusive(v.begin(),v.end(),0);
    std::cout<<std::fixed<<std::setprecision(6);
    std::cout<<num<<"\t\t";
    bool result = true;
    if(!vout.empty()) result = (endPoint == vout.at(vout.size()-1));
    if ((v1 == vout) && result ) {
      std::cout<<"PASS\t\t";
      std::cout<<input->getRunTime()<<std::endl;
    }
    else  {
        std::cout<<"FAIL\t\t";
        std::cout<<"NA"<<std::endl;
    }
  }
};
#endif
