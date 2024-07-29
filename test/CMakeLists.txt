cmake_minimum_required(VERSION 3.27)

enable_testing()

add_executable(
    test_transforms
    test_transforms.cpp
)
target_link_libraries(
    test_transforms
    fourier_mellin
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_transforms)

