#include "Maths.h"
__device__ double sigmoid(double x)
{
    if (x < -100.0)
        x = -100.0;
    if (x > 100.0)
        x = 100.0;
    return 1.0 / (1.0 + std::exp(-x));
}

__global__ void applySigmoidToVector(const double *vec1, double *h_resVec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        h_resVec[idx] = 0;
        sigmoid(vec1[idx]);
    }
}

double *executeSigmoidKernel(const double *vec, int cols, int rows)
{
    double *d_matResult;
    double *d_mat;
    size_t sizeT = cols * rows * sizeof(double);
    double *h_matResult = new double[cols * rows];
    cudaMalloc((void **)&d_mat, sizeT);
    cudaMalloc((void **)&d_matResult, sizeT);
    cudaMemcpy(d_mat, vec, sizeT, cudaMemcpyHostToDevice);
    int threads = 512;
    dim3 block(rows);
    applySigmoidToVector<<<threads, block>>>(d_mat, d_matResult, (rows * cols));

    cudaMemcpy(h_matResult, d_matResult, sizeT, cudaMemcpyDeviceToHost);
    cudaFree(d_matResult);
    cudaFree(d_mat);
    return h_matResult;
}