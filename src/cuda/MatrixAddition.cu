#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__global__ void CudaKernels::matrixAddition(const float *mat1, const float *mat2, float *matResult, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matResult[idx] = mat1[idx] + mat2[idx];
    }
}

float *CudaLaunchers::add(float *mat1, const float *mat2, float *matResult, size_t size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    CudaKernels::matrixAddition<<<blocks, threads>>>(mat1, mat2, matResult, size);
    cudaDeviceSynchronize();
    return matResult;
}