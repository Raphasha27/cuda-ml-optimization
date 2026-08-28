cmake_minimum_required(VERSION 3.28)
project(cuda_ml_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)

add_executable(matrix_multiply
    src/cuda/matrix_multiply.cu
    src/cpp/benchmark.cpp
)

target_include_directories(matrix_multiply PRIVATE include)
target_link_libraries(matrix_multiply PRIVATE ${CUDA_LIBRARIES})
