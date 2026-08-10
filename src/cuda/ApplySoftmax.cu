#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__global__ void applyVecSoftMaxKernel(const double *mat, double *matResult, int rows, int cols)
{
    int col = blockIdx.x;
    int row = threadIdx.x;

    if (col < cols && row < rows)
    {
        double max;
        max = mat[0 * cols + col];
        for (int i = 1; i < rows; i++) // bei eins, weil 0 ist schon uf max gesetztt
        {
            if (mat[i * cols + col] > max)
            {
                max = mat[i * cols + col];
            }
        }
        // e^(x - max)//Warum darf auf meiner GRAKA kein double benutzen
        __shared__ double sum;
        if (row == 0)
        {
            sum = 0.0;
            for (int i = 0; i < rows; i++)
            {
                sum += exp(mat[i * cols + col] - max);
            }
        }
        matResult[row * cols + col] = exp(mat[row * cols + col] - max) / sum;
    }
}
double *execcuteSoftMaxKernel(double *mat, int rows, int cols)
{
    double *d_matResult;
    double *d_mat;
    double *h_matRes = new double[cols * rows];
    size_t size = rows * cols * sizeof(double);
    cudaMalloc((void **)&d_mat, size);
    cudaMalloc((void **)&d_matResult, size);

    cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);
    dim3 grids(cols);
    dim3 block(rows);
    applyVecSoftMaxKernel<<<grids, block>>>(d_mat, d_matResult, rows, cols);
    cudaMemcpy(h_matRes, d_matResult, size, cudaMemcpyDeviceToHost);
    cudaFree(d_matResult);
    cudaFree(d_mat);
    return h_matRes;
}
__global__ void CudaKernels::softmaxKernel(float *input, float *output, int rows, int cols)
{
    int col = blockIdx.x;
    int row = threadIdx.x;

    if (col < cols && row < rows)
    {
        float max;
        max = input[0 * cols + col];
        for (int i = 1; i < rows; i++) // bei eins, weil 0 ist schon uf max gesetztt
        {
            if (input[i * cols + col] > max)
            {
                max = input[i * cols + col];
            }
        }
        // e^(x - max)//Warum darf auf meiner GRAKA kein double benutzen
        __shared__ float sum;
        if (row == 0)
        {
            sum = 0.0;
            for (int i = 0; i < rows; i++)
            {
                sum += exp(input[i * cols + col] - max);
            }
        }
        __syncthreads();
        output[row * cols + col] = exp(input[row * cols + col] - max) / sum;
    }
}

void CudaLaunchers::softmax(float *mat, float *matResult, int rows, int cols)
{
    dim3 grids(cols);
    dim3 block(rows);
    CudaKernels::softmaxKernel<<<grids, block>>>(mat, matResult, rows, cols);
    cudaDeviceSynchronize();
}