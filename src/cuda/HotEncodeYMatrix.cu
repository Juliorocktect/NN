#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

double *hotEncodeYMatrix(double *labels, int size)
{
    double *d_mat;
    double *d_mat_result;
    size_t sizeMat = size * sizeof(double);
    size_t sizeMatResult = size * 10 * sizeof(double);
    double *h_mat_result = new double[size * 10];
    cudaMalloc((void **)&d_mat, sizeMat);
    cudaMalloc((void **)&d_mat_result, sizeMatResult);
    cudaMemset(d_mat_result, 0, sizeMatResult);
    cudaMemcpy(d_mat, labels, sizeMat, cudaMemcpyHostToDevice);
    int threads = 10;
    dim3 block(size);
    applyHotEncodeToMatrix<<<threads, block>>>(d_mat, d_mat_result, size * 10);
    cudaMemcpy(h_mat_result, d_mat_result, sizeMatResult, cudaMemcpyDeviceToHost);
    cudaFree(d_mat);
    cudaFree(d_mat_result);
    return h_mat_result;
}
__global__ void applyHotEncodeToMatrix(double *mat, double *mat_result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        int label = static_cast<int>(mat[idx]);
        if (mat[idx] < 10)
        {
            mat_result[idx * 10 + label] = 1.0;
        }
    }
}

__global__ void CudaKernels::hotEncodeToMatrixKernel(float *mat, float *matResult, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float v = mat[idx];
        // Bei Float ist das wichtig so zu prüfen
        if (v >= 0.0f && v < 10.0f)
        {
            int label = static_cast<int>(v);
            matResult[idx * 10 + label] = 1.0f;
        }
    }
}
void CudaLaunchers::hotEncodeYMatrix(float *mat, float *matResult, size_t size)
{
    int threads = 10;
    dim3 block(size);
    CudaKernels::hotEncodeToMatrixKernel<<<threads, block>>>(mat, matResult, size);
    cudaDeviceSynchronize();
}