#include "exclusiveScan.h"
#include "utils.h"
Assignment::Assignment(){

  srand (time(NULL));
  N = 1*(rand() % 1000);
  fillNum();

}

Assignment::Assignment(int num){

  setNum(num);

}

Assignment::Assignment(int num,int* inNum){

  N = num;
  in.assign(inNum, inNum + N);

}

Assignment::Assignment(std::vector<int> data){

  N = data.size();
  in.assign(data.begin(), data.end());

}



void Assignment::fillNum(){

  srand (time(NULL));
  if(!in.empty()) {
    in.clear();
  }
  in.assign(N,0);
  for (int i = 0;i < N;i++){
    in[i] = rand() % 10;
  }

}

void Assignment::setNum(int num){

  N = num;
  fillNum();

}

void Assignment::setData(std::vector<int> inNum){

  N = inNum.size();
  in = inNum;

}


void Assignment::run(){

  long start_time = getSeconds();

  if(!out.empty()){ out.clear();}
  out.resize(in.size(),0);
  out.at(0) = 0;
  for (int i = 1; i < N; i++)
  {
    out[i] = out[i-1] + in[i-1];
  }

  long end_time = getSeconds();

  runTime =  (end_time - start_time);

}
