#ifndef UTILSH
#define UTILSH
#include <vector>
#include <iostream>
#include <iterator>
#include <chrono>
#include <algorithm>

void exclusive_scan(std::vector<int>::iterator inStart,std::vector<int>::iterator inEnd, std::vector<int>::iterator outStart,int Start);
int sumExclusive(std::vector<int>::iterator inStart,std::vector<int>::iterator inEnd,int Start);

long getSeconds();


#endif
