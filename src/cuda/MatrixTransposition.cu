#include "Maths.h"


double* executeMatrixTransposition(const double* mat1, int rows, int cols)
{
    double* d_mat1;
    double* d_matResult;
    size_t size = rows * cols * sizeof(double);
    double* h_matResult = new double[size];
    cudaMalloc((void**)&d_mat1, size);
    cudaMalloc((void**)&d_matResult, size);

    cudaMemcpy(d_mat1, mat1, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x, (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transposeMatrix << <threadsPerBlock, blocksPerGrid >> > (d_mat1, d_matResult, rows, cols, cols, rows);

    cudaMemcpy(h_matResult, d_matResult, size, cudaMemcpyDeviceToHost);
    cudaFree(&d_mat1);
    cudaFree(&d_matResult);
    return h_matResult;

}
__global__ void transposeMatrix(const double* mat1, double* matResult, int rows1, int cols1, int rows2, int cols2)
{
    int cols = blockIdx.x * blockDim.x + threadIdx.x;
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    if (cols < cols1 && rows < rows2)
    {
        matResult[cols + cols1 * rows] = mat1[cols * rows2 + rows];
    }
}