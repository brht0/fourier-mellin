cmake_minimum_required(VERSION 3.14)

# TODO: Better way to depend on opencv?

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)
