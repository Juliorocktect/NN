#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

float *CudaLaunchers::vectorAddition(const float *vec1, const float *vec2, float *vecResult, size_t size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    CudaKernels::vectorAddition<<<blocks, threads>>>(vec1, vec2, vecResult, size);
    cudaDeviceSynchronize();
    return vecResult;
}

__global__ void CudaKernels::vectorAddition(const float *vec1, const float *vec2, float *vecResult, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        vecResult[index] = vec1[index] + vec2[index];
    }
}