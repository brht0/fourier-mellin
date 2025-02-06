cmake_minimum_required(VERSION 3.27)

enable_testing()

add_executable(
    test_transforms
    test_transforms.cpp
)
target_link_libraries(
    test_transforms
    fourier-mellin-library
    GTest::gtest_main
)

add_executable(
    test_fourier_mellin
    test_fourier_mellin.cpp
)
target_link_libraries(
    test_fourier_mellin
    fourier-mellin-library
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_transforms)
gtest_discover_tests(test_fourier_mellin)

