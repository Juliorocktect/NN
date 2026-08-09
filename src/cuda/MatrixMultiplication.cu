#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

double *executeMatrixMultiplicationKernel(const double *mat1, const double *mat2, const int rows1, const int cols1, const int rows2, const int cols2)
{
    if (cols1 != rows2)
    {
        std::cerr << "Matrixmultiplikation nicht möglich: falsche Dimensionen!" << std::endl;
        return nullptr;
    }
    size_t size1 = rows1 * cols1 * sizeof(double);
    size_t size2 = rows2 * cols2 * sizeof(double);
    size_t sizeResult = rows1 * cols2 * sizeof(double);
    double *h_matResult = new double[rows1 * cols2];

    double *d_mat1, *d_mat2, *d_matResult;
    cudaMalloc((void **)&d_mat1, size1);
    cudaMalloc((void **)&d_mat2, size2);
    cudaMalloc((void **)&d_matResult, sizeResult);

    cudaMemcpy(d_mat1, mat1, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, size2, cudaMemcpyHostToDevice);

    // dimensionen festlegen
    dim3 threadsPerBlock(16, 16); // 256
    dim3 blocksPerGrid((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_mat1, d_mat2, d_matResult, rows1, cols1, cols2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_matResult, d_matResult, sizeResult, cudaMemcpyDeviceToHost);

    // Man muss die Anordnung der matrix wiederherstellen
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_matResult);
    return h_matResult;
}

__global__ void matrixMultiplication(const double *mat1, const double *mat2, double *mat3, int rows1, int cols1, int cols2)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows1 && col < cols2)
    {
        double sum = 0.0;
        for (int i = 0; i < cols1; ++i)
        {
            sum += mat1[row * cols1 + i] * mat2[i * cols2 + col];
        }
        mat3[row * cols2 + col] = sum;
    }
}
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