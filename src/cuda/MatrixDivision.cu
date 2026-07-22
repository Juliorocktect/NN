#include "Maths.h"

double* executeMatrixDivision(double* mat1, double dividend, int rows, int cols)
{
    double* d_mat1;
    double* d_mat_result;
    size_t sizeMat = rows * cols * sizeof(double);
    cudaMalloc((void**)&d_mat1, sizeMat);
    cudaMalloc((void**)&d_mat_result, sizeMat);
    cudaMemcpy(d_mat1, mat1, sizeMat, cudaMemcpyHostToDevice);
    dim3 grid(cols);
    dim3 block(rows);
    applyMatrixDivision << <grid, block >> > (d_mat1, d_mat_result, dividend, rows, cols);
    double* h_res = new double[rows * cols];
    cudaMemcpy(h_res, d_mat_result, sizeMat, cudaMemcpyDeviceToHost);
    cudaFree(d_mat1);
    cudaFree(d_mat_result);
    return h_res;
}
__global__ void applyMatrixDivision(double* mat, double* mat_result, double div, int rows, int cols)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row < rows && col < cols)
    {
        mat_result[row * cols + col] = mat[row * cols + col] / div;
    }
}