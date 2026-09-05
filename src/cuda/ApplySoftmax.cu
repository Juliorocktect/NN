#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__global__ void CudaKernels::softmaxKernel(float *input, float *output, int rows, int cols)
{
    int col = blockIdx.x;
    int row = blockIdx.y * blockDim.x + threadIdx.x;

    if (col < cols && row < rows)
    {
        float maxValue = input[col];
        for (int i = 1; i < rows; i++) // bei eins, weil 0 ist schon uf max gesetztt
        {
            if (input[i * cols + col] > maxValue)
            {
                maxValue = input[i * cols + col];
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < rows; i++)
        {
            sum += expf(input[i * cols + col] - maxValue);
        }

        output[row * cols + col] = expf(input[row * cols + col] - maxValue) / sum;
    }
}

void CudaLaunchers::softmax(float *mat, float *matResult, int rows, int cols)
{
    if (mat == nullptr || matResult == nullptr || rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Invalid softmax input or dimensions");
    }

    constexpr int threadsPerBlock = 256;
    dim3 grids(cols, (rows + threadsPerBlock - 1) / threadsPerBlock);
    dim3 block(threadsPerBlock);
    CudaKernels::softmaxKernel<<<grids, block>>>(mat, matResult, rows, cols);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}