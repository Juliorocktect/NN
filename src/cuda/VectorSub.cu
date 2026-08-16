#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

void CudaLaunchers::vectorSubtraction(const float *vec1, const float *vec2, float *vecResult, size_t size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    CudaKernels::vectorSubtraction<<<blocks, threads>>>(vec1, vec2, vecResult, size);
    cudaDeviceSynchronize();
}

__global__ void CudaKernels::vectorSubtraction(const float *vec1, const float *vec2, float *vecResult, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        vecResult[index] = vec1[index] - vec2[index];
    }
}