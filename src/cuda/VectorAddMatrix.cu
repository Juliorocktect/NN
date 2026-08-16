#include "CudaLaunchers.cuh"
#include "CudaKernels.cuh"

__global__ void CudaKernels::vectorAddMatrixKernel(float *mat, float *vec, float *mat_result, size_t sizeVec, size_t rows, size_t cols)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row < rows && col < cols)
    {
        mat_result[row * cols + col] = mat[row * cols + col] + vec[row];
    }
}
void CudaLaunchers::vectorAddMatrix(float *mat, float *vec, float *matResult, size_t sizeVec, size_t rows, size_t cols)
{
    dim3 grid(cols);
    dim3 block(rows);
    CudaKernels::vectorAddMatrixKernel<<<grid, block>>>(mat, vec, matResult, sizeVec, rows, cols);
    cudaDeviceSynchronize();
}
