#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__global__ void CudaKernels::argmax(const float *mat, int rows, int cols, float *result)
{
    int col = blockIdx.x;
    result[col] = 0;
    float maxVal = mat[col];
    int index = 0;
    for (int i = 0; i < rows; i++)
    {
        float v = mat[i * cols + col];
        if (col == 0)
        {
            printf("Value: %f,i: %d, col: %d\n", v, i, col);
        }

        if (v > maxVal)
        {
            maxVal = v;
            result[col] = i;
        }
    }
}

void CudaLaunchers::argmax(const float *mat, int rows, int cols, float *result)
{
    int threadsPerBlock = 1;
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;
    CudaKernels::argmax<<<blocksPerGrid, threadsPerBlock>>>(mat, rows, cols, result);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}