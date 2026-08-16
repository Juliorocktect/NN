#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__global__ void CudaKernels::matrixMultiplication(const float *mat1, const float *mat2, float *mat3, int rows1, int cols1, int cols2)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows1 && col < cols2)
    {
        float sum = 0.0f;
        for (int i = 0; i < cols1; ++i)
        {
            sum += mat1[row * cols1 + i] * mat2[i * cols2 + col];
        }
        mat3[row * cols2 + col] = sum;
    }
}

float *CudaLaunchers::multiply(float *mat1, const float *mat2, float *matResult, int rows1, int cols1, int cols2)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    CudaKernels::matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(mat1, mat2, matResult, rows1, cols1, cols2);
    cudaDeviceSynchronize();
    return matResult;
}