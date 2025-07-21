cmake_minimum_required(VERSION 3.10)
project(OpenCVProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

# Find OpenCV and Ceres
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# Find glm (header-only, lightweight)
find_package(glm REQUIRED)

# Add include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
)

# Add subdirectory for tests
enable_testing()

# Include local headers
include_directories(include)

# Add source files to aux library
set(AUX_SOURCES 
    src/scaleSpace.cpp
    src/dog.cpp
    src/keypointDetection.cpp
    src/refine.cpp
    src/histogram.cpp
    src/descriptor.cpp
    src/visualization.cpp
)

add_library(aux STATIC ${AUX_SOURCES})
target_include_directories(aux PUBLIC include)
target_link_libraries(aux ${OpenCV_LIBS})

add_executable(main main.cpp)
target_link_libraries(main aux ${OpenCV_LIBS})

# Add tests
add_subdirectory(tests)

# Set CUDA compiler and enable CUDA language
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
# enable_language(CUDA)
