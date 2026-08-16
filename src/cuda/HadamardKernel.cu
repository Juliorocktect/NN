#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

void CudaLaunchers::hadamardProduct(float *mat1, const float *mat2, float *matResult, size_t size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    CudaKernels::hadamardProduct<<<blocks, threads>>>(mat1, mat2, matResult, size);
    cudaDeviceSynchronize();
}
__global__ void CudaKernels::hadamardProduct(float *mat1, const float *mat2, float *matResult, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matResult[idx] = mat1[idx] * mat2[idx];
    }
}