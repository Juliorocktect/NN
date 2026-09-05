#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__global__ void CudaKernels::matrixDivision(const float *mat1, double v, float *matResult, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matResult[idx] = mat1[idx] / v;
    }
}
float *CudaLaunchers::divide(const float *mat1, double v, float *matResult, size_t size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    CudaKernels::matrixDivision<<<blocks, threads>>>(mat1, v, matResult, size);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    return matResult;
}