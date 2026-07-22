#include "Maths.h"

__global__ void vectorSubKernel(const double* vec1, const double* vec2, double* vecRes, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        vecRes[idx] = vec1[idx] - vec2[idx];
    }
}

double* executeVecSubKernel(double* vec1, double* vec2, int size)
{
    double* d_vec1;
    double* d_vec2;
    double* d_vec_res;
    double* h_res = new double[size];
    size_t sizeVec = size * sizeof(double);
    cudaMalloc((void**)&d_vec1, sizeVec);
    cudaMalloc((void**)&d_vec2, sizeVec);
    cudaMalloc((void**)&d_vec_res, sizeVec);

    cudaMemcpy(d_vec1, vec1, sizeVec, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, vec2, sizeVec, cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    vectorSubKernel << <threadsPerBlock, blocksPerGrid >> > (d_vec1, d_vec2, d_vec_res, size);

    cudaMemcpy(h_res, d_vec_res, sizeVec, cudaMemcpyDeviceToHost);
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_vec_res);
    return h_res;
}