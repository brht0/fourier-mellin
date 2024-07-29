cmake_minimum_required(VERSION 3.27)

project(fourier-mellin)

option(BUILD_PYTHON_MODULE "Build the Python module" OFF)

set(MODULE_NAME cv2_fourier_mellin)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")

add_subdirectory(ext)
add_subdirectory(src)
add_subdirectory(test)
