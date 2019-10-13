cmake_minimum_required(VERSION 2.8.9)
project(SequentialRun)

include_directories(include)

file(GLOB SOURCES "test/*.cpp" "src/*.cpp")

add_executable(prefixScan ${SOURCES})
