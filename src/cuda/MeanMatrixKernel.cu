#include "Maths.h"

double* executeMeanMatrixKernel(double* mat, int rows, int cols)
{
    double* d_mat_result;
    double* d_mat;
    size_t size = rows * cols * sizeof(double);
    int n = rows * cols;
    double* h_res = new double[n];
    cudaMalloc((void**)&d_mat, size);
    cudaMalloc((void**)&d_mat_result, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    meanMatrixKernel << <threadsPerBlock, blocksPerGrid >> > (d_mat, d_mat_result, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res, d_mat_result, size, cudaMemcpyDeviceToHost);

    cudaFree(&d_mat_result);
    cudaFree(&d_mat);
    return h_res;

}
__global__ void meanMatrixKernel(const double* mat, double* resultMatrix, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        double sum = 0.0;
        for (int col = 0; col < cols; ++col)
        {
            sum += mat[row * cols + col];
        }
        resultMatrix[row] = sum / cols;
    }
}