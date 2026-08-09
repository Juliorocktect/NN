#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

double *executeMatrixDivision(double *mat1, double dividend, int rows, int cols)
{
    double *d_mat1;
    double *d_mat_result;
    size_t sizeMat = rows * cols * sizeof(double);
    cudaMalloc((void **)&d_mat1, sizeMat);
    cudaMalloc((void **)&d_mat_result, sizeMat);
    cudaMemcpy(d_mat1, mat1, sizeMat, cudaMemcpyHostToDevice);
    dim3 grid(cols);
    dim3 block(rows);
    applyMatrixDivision<<<grid, block>>>(d_mat1, d_mat_result, dividend, rows, cols);
    double *h_res = new double[rows * cols];
    cudaMemcpy(h_res, d_mat_result, sizeMat, cudaMemcpyDeviceToHost);
    cudaFree(d_mat1);
    cudaFree(d_mat_result);
    return h_res;
}
__global__ void applyMatrixDivision(double *mat, double *mat_result, double div, int rows, int cols)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row < rows && col < cols)
    {
        mat_result[row * cols + col] = mat[row * cols + col] / div;
    }
}
__global__ void CudaKernels::matrixDivision(const float *mat1, double v, float *matResult, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matResult[idx] = mat1[idx] / v;
    }
}
float *CudaLaunchers::divide(const float *mat1, double v, float *matResult, size_t size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    CudaKernels::matrixDivision<<<blocks, threads>>>(mat1, v, matResult, size);
    cudaDeviceSynchronize();
    return matResult;
}