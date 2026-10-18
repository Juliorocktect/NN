#include "Maths.h"
#include "CudaLaunchers.cuh"
#include "CudaKernels.cuh"

__device__ double sigmoid(double x)
{
    if (x < -100.0)
        x = -100.0;
    if (x > 100.0)
        x = 100.0;
    return 1.0 / (1.0 + std::exp(-x));
}

void CudaLaunchers::sigmoid(float *mat, float *matResult, size_t size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    CudaKernels::sigmoidKernel<<<blocks, threads>>>(mat, matResult, size);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
__global__ void CudaKernels::sigmoidKernel(float *mat, float *matResult, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matResult[idx] = CudaKernels::sigmoidFunction(mat[idx]);
    }
}
__device__ float CudaKernels::sigmoidFunction(float x)
{
    if (x < -100.0f)
        x = -100.0f;
    if (x > 100.0f)
        x = 100.0f;
    return 1.0f / (1.0f + expf(-x));
}