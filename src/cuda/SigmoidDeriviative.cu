#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__device__ float CudaKernels::sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ float CudaKernels::sigmoidDeriviativeFunction(float x)
{
    float y = CudaKernels::sigmoid(x);
    return y * (1 - y);
}

float *CudaLaunchers::sigmoidDeriviative(float *input, float *output, size_t size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    CudaKernels::sigmoidDeriviative<<<blocksPerGrid, threadsPerBlock>>>(input, output, size);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    return output;
}

__global__ void CudaKernels::sigmoidDeriviative(float *input, float *output, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = CudaKernels::sigmoidDeriviativeFunction(input[idx]);
    }
}