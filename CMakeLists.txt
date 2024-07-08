cmake_minimum_required(VERSION 3.14)

project(fourier-mellin)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")

# add_subdirectory(ext)
add_subdirectory(src)
