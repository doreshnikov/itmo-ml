cmake_minimum_required(VERSION 3.10)
project(itmo-ml)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(FNAME="0.40_0.65.txt")
add_compile_definitions(LOCAL)
#add_compile_definitions(UNTIMED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")

add_executable(gradient lab02_linear/gradient.cpp)
add_executable(gradient_old lab02_linear/gradient_old.cpp)