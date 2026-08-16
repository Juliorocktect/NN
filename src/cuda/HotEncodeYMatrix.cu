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
    int threads = 10;
    dim3 block(size);
    CudaKernels::hotEncodeToMatrixKernel<<<threads, block>>>(mat, matResult, size);
    cudaDeviceSynchronize();
}