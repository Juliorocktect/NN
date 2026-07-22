#include "Maths.h"

__global__ void singleVMatrixMultiplyKernel(double* mat, double* mat_res, double v, int rows, int cols)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row < rows && col < cols)
    {
        mat_res[row * cols + col] = mat[row * cols + col] * v;
    }
}
double* executeSingleVMatrixMultiplication(double* mat, double v, int rows, int cols)
{
    double* d_mat1;
    double* d_mat_result;
    size_t sizeMat = rows * cols * sizeof(double);
    cudaMalloc((void**)&d_mat1, sizeMat);
    cudaMalloc((void**)&d_mat_result, sizeMat);
    cudaMemcpy(d_mat1, mat, sizeMat, cudaMemcpyHostToDevice);
    dim3 grid(cols);
    dim3 block(rows);
    singleVMatrixMultiplyKernel << <grid, block >> > (d_mat1, d_mat_result, v, rows, cols);
    double* h_res = new double[rows * cols];
    cudaMemcpy(h_res, d_mat_result, sizeMat, cudaMemcpyDeviceToHost);
    cudaFree(d_mat1);
    cudaFree(d_mat_result);
    return h_res;
}