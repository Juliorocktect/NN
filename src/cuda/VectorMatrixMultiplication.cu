#include "CudaLaunchers.cuh"
#include "CudaKernels.cuh"

__global__ void CudaKernels::matrixMultiplicationWithFloatKernel(float *mat, float value, float *matResult, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matResult[idx] = mat[idx] * value;
    }
}

void CudaLaunchers::matrixMultiplicationWithFloat(float *mat, float value, float *matResult, size_t rows, size_t cols)
{
    dim3 grid(cols);
    dim3 block(rows);
    size_t size = rows * cols;
    CudaKernels::matrixMultiplicationWithFloatKernel<<<grid, block>>>(mat, value, matResult, size);
    cudaDeviceSynchronize();
}
