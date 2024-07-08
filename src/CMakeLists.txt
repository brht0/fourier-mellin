cmake_minimum_required(VERSION 3.14)

find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)

set(SOURCES fft.cpp filters.cpp fourier_mellin.cpp log_polar_map.cpp)

add_executable(${PROJECT_NAME} main.cpp ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

# set(MODULE_NAME brht_fourier_mellin)
set(MODULE_NAME cv2_fourier_mellin)
add_definitions(-DMODULE_NAME=${MODULE_NAME})
pybind11_add_module("${MODULE_NAME}" fourier_mellin_module.cpp ${SOURCES})
target_link_libraries("${MODULE_NAME}" PRIVATE ${OpenCV_LIBS})
