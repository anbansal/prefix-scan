#include "utils.h"

void exclusive_scan(std::vector<int>::iterator inStart,std::vector<int>::iterator inEnd, std::vector<int>::iterator outStart,int Start){
  if(inStart != inEnd){
    *outStart = Start;
    outStart++;
    for(;inStart<inEnd-1;inStart++){
      *outStart = *inStart + *(outStart-1);
      outStart++;
    }
  }
}
int sumExclusive(std::vector<int>::iterator inStart,std::vector<int>::iterator inEnd,int Start){
  int out = Start;
  if(inStart != inEnd){
    for(;inStart<inEnd-1;inStart++){
      out += *inStart;
    }
}
  return out;
}

long getSeconds() {
  std::chrono::high_resolution_clock m_clock;
  return (long)std::chrono::duration_cast<std::chrono::nanoseconds>
            (m_clock.now().time_since_epoch()).count();
}
