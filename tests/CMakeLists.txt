# Download and add GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

# Test executable
add_executable(test_aux
    ${CMAKE_CURRENT_SOURCE_DIR}/test_scaleSpace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_keypointDetection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_refine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_histogram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_visualization.cpp
)

# Link against the main library and GTest
target_link_libraries(test_aux
    gtest
    gtest_main
    ${OpenCV_LIBS}
    aux
)

# Add test discovery
include(GoogleTest)
gtest_discover_tests(test_aux)
