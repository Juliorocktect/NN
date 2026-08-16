#include "CudaLaunchers.cuh"
#include "CudaKernels.cuh"

__global__ void CudaKernels::vectorValueMultiplication(float *vec, float *vecResult, size_t size, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        vecResult[idx] = vec[idx] * value;
    }
}

void CudaLaunchers::vectorValueMultiplication(float *vec, float *vecResult, float value, size_t size)
{
    int threadsPerBlock = 128;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    CudaKernels::vectorValueMultiplication<<<threadsPerBlock, blocksPerGrid>>>(vec, vecResult, size, value);
}