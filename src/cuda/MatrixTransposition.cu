#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__global__ void CudaKernels::transposeMatrixKernel(const float *input, float *output, int rows1, int cols1, int rows2, int cols2)
{
    int cols = blockIdx.x * blockDim.x + threadIdx.x;
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    if (cols < cols2 && rows < rows2)
    {
        output[rows * cols2 + cols] = input[cols * cols1 + rows];
    }
}

float *CudaLaunchers::transpose(const float *mat, float *matResult, int rows, int cols)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    CudaKernels::transposeMatrixKernel<<<blocksPerGrid, threadsPerBlock>>>(mat, matResult, rows, cols, cols, rows);
    cudaDeviceSynchronize();
    return matResult;
}