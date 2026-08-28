#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

__global__ void matrixMultiply(const float* A, const float* B, float* C, int N);
__global__ void reluActivation(float* data, int size);
__global__ void sigmoidActivation(float* data, int size);

#endif
