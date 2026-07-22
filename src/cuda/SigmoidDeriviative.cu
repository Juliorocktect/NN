#include "Maths.h"
__global__ void applySigmoidDeriviative(const double* mat1, double* mat_result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        mat_result[idx] = sigmoidDeriviative(mat1[idx]);
    }
}
double* executeSigmoidDeriviativeKernel(const double* mat1, int rows, int cols)
{
    double* d_mat_result;
    double* d_mat1;
    size_t n = rows * cols * sizeof(double);
    int o = rows * cols;
    double* h_result = new double[o];
    cudaMalloc((void**)&d_mat_result, n);
    cudaMalloc((void**)&d_mat1, n);
    cudaMemcpy(d_mat1, mat1, n, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (o + threadsPerBlock - 1) / threadsPerBlock;
    applySigmoidDeriviative << <threadsPerBlock, blocksPerGrid >> > (d_mat1, d_mat_result, o);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_mat1, n, cudaMemcpyDeviceToHost);
    cudaFree(&d_mat_result);
    cudaFree(&d_mat1);
    return h_result;
}