#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__global__ void CudaKernels::meanMatrixRowise(float *mat, float *matResult, size_t rows, size_t cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows)
    {
        float sum = 0.0;
        for (int col = 0; col < cols; ++col)
        {
            sum += mat[idx * cols + col];
        }
        matResult[idx] = sum / cols;
    }
}

void CudaLaunchers::meanMatrixRowise(float *mat, float *matResult, size_t rows, size_t cols)
{
    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    CudaKernels::meanMatrixRowise<<<blocksPerGrid, threadsPerBlock>>>(mat, matResult, rows, cols);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}