cmake_minimum_required(VERSION 3.14)

find_package(OpenCV REQUIRED)

set(SOURCES main.cpp fft.cpp filters.cpp fourier_mellin.cpp log_polar_map.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
