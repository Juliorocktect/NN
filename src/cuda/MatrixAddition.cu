#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

double *executeMatrixAdditionKernel(const double *mat1, const double *mat2, const int rows1, const int cols1, int rows2, int cols2)
{
    if (rows1 != rows2 || cols1 != cols2)
    {
        std::cerr << "Matrixaddition nicht möglich, falsche Dimensionen!" << std::endl;
        return nullptr;
    }
    double *d_mat1;
    double *d_mat2;
    double *d_matResult;
    size_t sizeMat = rows1 * cols1 * sizeof(double);
    double *h_result = new double[sizeMat];
    cudaMalloc((void **)&d_mat1, sizeMat);
    cudaMalloc((void **)&d_mat2, sizeMat);
    cudaMalloc((void **)&d_matResult, sizeMat);

    cudaMemcpy(d_mat1, mat1, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, sizeMat, cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksPerGrid = (sizeMat + threadsPerBlock - 1) / threadsPerBlock;

    MatrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_mat1, d_mat2, d_matResult, sizeMat);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_matResult, sizeMat, cudaMemcpyDeviceToHost);
    cudaFree(&d_mat1);
    cudaFree(&d_mat2);
    cudaFree(&d_matResult);
    return h_result;
}

__global__ void MatrixAdd(const double *mat1, const double *mat2, double *matResult, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matResult[idx] = mat1[idx] + mat2[idx];
    }
}
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