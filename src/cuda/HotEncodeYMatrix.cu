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
            matResult[label * size + idx] = 1.0f;
        }
    }
}
void CudaLaunchers::hotEncodeYMatrix(float *mat, float *matResult, size_t size)
{
    if (mat == nullptr || matResult == nullptr)
        throw std::invalid_argument("Hot encoding requires valid device buffers");
    if (size == 0)
        return;

    cudaError_t error = cudaMemset(matResult, 0, size * 10 * sizeof(float));
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    CudaKernels::hotEncodeToMatrixKernel<<<blocksPerGrid, threadsPerBlock>>>(mat, matResult, size);

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}