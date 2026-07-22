#include "Maths.h"

__global__ void vectorAddMatrixKernel(double* mat, double* vec, double* mat_result, int sizeVec, int rows, int cols)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row < rows && col < cols)
    {
        mat_result[row * cols + col] = mat[row * cols + col] + vec[row];
    }

}
double* vectorAddMatrix(double* mat, double* vec, int sizeVec, int rows, int cols)
{
    double* d_mat;
    double* d_mat_result;
    double* d_vec;
    size_t sizeMat = rows * cols * sizeof(double);
    size_t sizeTVec = sizeVec * sizeof(double);
    double* h_mat_result = new double[rows * cols];
    cudaMalloc((void**)&d_mat, sizeMat);
    cudaMalloc((void**)&d_mat_result, sizeMat);
    cudaMalloc((void**)&d_vec, sizeTVec);
    cudaMemcpy(d_mat, mat, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec, sizeTVec, cudaMemcpyHostToDevice);
    dim3 grid(cols);
    dim3 block(rows);
    vectorAddMatrixKernel << <grid, block >> > (d_mat, d_vec, d_mat_result, sizeVec, rows, cols);
    cudaMemcpy(h_mat_result, d_mat_result, sizeMat, cudaMemcpyDeviceToHost);
    cudaFree(d_mat);
    cudaFree(d_mat_result);
    cudaFree(d_vec);
    return h_mat_result;
}