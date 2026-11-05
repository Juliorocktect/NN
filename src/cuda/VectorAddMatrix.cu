#include "CudaLaunchers.cuh"
#include "CudaKernels.cuh"

__global__ void CudaKernels::vectorAddMatrixKernel(float *mat, float *vec, float *mat_result, size_t sizeVec, size_t rows, size_t cols)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t rowCount = rows < sizeVec ? rows : sizeVec;
    const size_t total = rowCount * cols;
    if (index < total)
    {
        const size_t row = index / cols;
        mat_result[index] = mat[index] + vec[row];
    }
}
void CudaLaunchers::vectorAddMatrix(float *mat, float *vec, float *matResult, size_t sizeVec, size_t rows, size_t cols)
{
    constexpr int threadsPerBlock = 256;
    const size_t total = rows * cols;
    const size_t blocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(static_cast<unsigned int>(blocks));
    dim3 block(threadsPerBlock);
    CudaKernels::vectorAddMatrixKernel<<<grid, block>>>(mat, vec, matResult, sizeVec, rows, cols);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
