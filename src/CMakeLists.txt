cmake_minimum_required(VERSION 3.14)

find_package(OpenCV REQUIRED)

set(SOURCES fourier_mellin.cpp utilities.cpp transform.cpp)

add_executable(demo demo.cpp ${SOURCES})
target_include_directories(demo PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(demo ${OpenCV_LIBS})

add_executable(benchmark benchmark.cpp ${SOURCES})
target_include_directories(benchmark PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(benchmark ${OpenCV_LIBS})

if(BUILD_PYTHON_MODULE)
    include_directories(${OpenCV_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

    add_definitions(-DMODULE_NAME=${MODULE_NAME})
    pybind11_add_module(${MODULE_NAME} fourier_mellin_module.cpp ${SOURCES})
    target_link_libraries(${MODULE_NAME} PRIVATE ${OpenCV_LIBS})
    set_target_properties(${MODULE_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${MODULE_NAME}
    )
endif()
