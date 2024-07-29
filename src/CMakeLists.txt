cmake_minimum_required(VERSION 3.27)

find_package(OpenCV REQUIRED)

set(SOURCES fourier_mellin.cpp utilities.cpp transform.cpp)

add_library(fourier_mellin STATIC
    ${SOURCES}
)
target_include_directories(fourier_mellin PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fourier_mellin ${OpenCV_LIBS})

add_executable(demo demo.cpp)
target_link_libraries(demo fourier_mellin)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark fourier_mellin)

if(BUILD_PYTHON_MODULE)
    include_directories(${OpenCV_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

    add_definitions(-DMODULE_NAME=${MODULE_NAME})
    pybind11_add_module(${MODULE_NAME} fourier_mellin_module.cpp ${SOURCES})
    target_link_libraries(${MODULE_NAME} PRIVATE ${OpenCV_LIBS})
    set_target_properties(${MODULE_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${MODULE_NAME}
    )
endif()
