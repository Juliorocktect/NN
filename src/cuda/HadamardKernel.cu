#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__global__ void hadamardKernel(double *mat1, double *mat2, double *mat_result, int rows, int cols)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row < rows && col < cols)
    {
        mat_result[row * cols + col] = mat1[row * cols + col] * mat2[row * cols + col];
    }
}
double *executeHadamardKernel(double *mat, double *mat2, int row1, int col1, int row2, int col2)
{
    if (row1 != row2 || col1 != col2)
    {
        std::cerr << "Matrixsubtraktion nicht möglich, falsche Dimensionen!" << std::endl;
        return nullptr;
    }
    double *d_mat1;
    double *d_mat2;
    double *d_mat_result;
    size_t sizeMat = row1 * col1 * sizeof(double);
    cudaMalloc((void **)&d_mat1, sizeMat);
    cudaMalloc((void **)&d_mat2, sizeMat);
    cudaMalloc((void **)&d_mat_result, sizeMat);
    cudaMemcpy(d_mat1, mat, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, sizeMat, cudaMemcpyHostToDevice);
    dim3 grid(col1);
    dim3 block(row1);
    hadamardKernel<<<grid, block>>>(d_mat1, d_mat2, d_mat_result, row1, col1);
    double *h_res = new double[row1 * col1];
    cudaMemcpy(h_res, d_mat_result, sizeMat, cudaMemcpyDeviceToHost);
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_mat_result);
    return h_res;
}

void CudaLaunchers::hadamardProduct(float *mat1, const float *mat2, float *matResult, size_t size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    CudaKernels::hadamardProduct<<<blocks, threads>>>(mat1, mat2, matResult, size);
    cudaDeviceSynchronize();
}
__global__ void CudaKernels::hadamardProduct(float *mat1, const float *mat2, float *matResult, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matResult[idx] = mat1[idx] * mat2[idx];
    }
}