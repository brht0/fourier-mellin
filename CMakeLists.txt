cmake_minimum_required(VERSION 3.27)

project(fourier-mellin-library LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MODULE_NAME fourier_mellin)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(ext)
add_subdirectory(src)
add_subdirectory(test)
