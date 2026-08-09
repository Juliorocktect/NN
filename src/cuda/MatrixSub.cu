#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

double *matrixSub(double *mat, double *mat2, int rows1, int cols1, int rows2, int cols2)
{
    if (rows1 != rows2 || cols1 != cols2)
    {
        std::cerr << "Matrixsubtraktion nicht möglich, falsche Dimensionen!" << std::endl;
        return nullptr;
    }
    double *d_mat1;
    double *d_mat2;
    double *d_mat_result;
    size_t sizeMat = rows1 * cols1 * sizeof(double);
    cudaMalloc((void **)&d_mat1, sizeMat);
    cudaMalloc((void **)&d_mat2, sizeMat);
    cudaMalloc((void **)&d_mat_result, sizeMat);
    cudaMemcpy(d_mat1, mat, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, sizeMat, cudaMemcpyHostToDevice);
    dim3 grid(cols1);
    dim3 block(rows1);
    matrixSubKernel<<<grid, block>>>(d_mat1, d_mat2, d_mat_result, rows1, cols1);
    double *h_res = new double[rows1 * cols1];
    cudaMemcpy(h_res, d_mat_result, sizeMat, cudaMemcpyDeviceToHost);
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_mat_result);
    return h_res;
}
__global__ void matrixSubKernel(double *mat1, double *mat2, double *matRes, int cols, int rows)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row < rows && col < cols)
    {
        matRes[row * cols + col] = mat1[row * cols + col] - mat2[row * cols + col];
    }
}

__global__ void CudaKernels::matrixSubstraction(const float *mat1, const float *mat2, float *matResult, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matResult[idx] = mat1[idx] - mat2[idx];
    }
}

float *CudaLaunchers::subtract(float *mat1, const float *mat2, float *matResult, size_t size)
{

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    CudaKernels::matrixSubstraction<<<blocks, threads>>>(mat1, mat2, matResult, size);
    cudaDeviceSynchronize();
    return matResult;
}