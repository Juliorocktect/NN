#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"
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
    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    CudaKernels::hotEncodeToMatrixKernel<<<blocksPerGrid, threadsPerBlock>>>(mat, matResult, size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}