cmake_minimum_required(VERSION 2.8.9)
project(ParallelRun)
find_package(CUDA QUIET REQUIRED)
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_35,code=compute_35 -std=c++11 -w
    )
include_directories(include)


file(GLOB SOURCES "test/*.cpp" "test/*.cu" "src/*.cpp" "src/*.cu")

cuda_add_executable(prefixScan ${SOURCES})
