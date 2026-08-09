#include "Maths.h"
#include "CudaLaunchers.cuh"
#include "CudaKernels.cuh"

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

void CudaLaunchers::sigmoid(float *mat, float *matResult, size_t size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    CudaKernels::sigmoidKernel<<<blocks, threads>>>(mat, matResult, size);
    cudaDeviceSynchronize();
}
__global__ void CudaKernels::sigmoidKernel(float *mat, float *matResult, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matResult[idx] = CudaKernels::sigmoidFunction(mat[idx]);
    }
}
__device__ float CudaKernels::sigmoidFunction(float x)
{
    if (x < -100.0f)
        x = -100.0f;
    if (x > 100.0f)
        x = 100.0f;
    return 1.0f / (1.0f + expf(-x));
}