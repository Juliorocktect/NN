#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

void CudaLaunchers::convolutionFirstLayer(const float *input, const float *kernel, float *output, size_t inputRows, size_t inputCols, size_t kernelRows, size_t kernelCols)
{
    int threads = 256;
    const short inputSize = inputRows * inputCols;
    const short kernelSize = kernelRows * kernelCols;
    if (inputSize % kernelSize != 0)
    {
        throw std::runtime_error("Input size must be a multiple of kernel size.");
    }
    const unsigned short stride = 1;
    const short sizeFeatureMap = (inputRows - kernelRows) / stride + 1;
    int blocks = (sizeFeatureMap + threads - 1) / threads;
    CudaKernels::convolutionKernel1<<<blocks, threads>>>(input, kernel, output, inputRows, inputCols, kernelRows, kernelCols);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

__global__ void CudaKernels::convolutionKernel1(const float *input, const float *kernel, float *output, size_t inputRows, size_t inputCols, size_t kernelRows, size_t kernelCols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < inputRows * inputCols)
    {
        const float *matRes;
        float* matKernelPart = new float[kernelRows * kernelCols];
        cudaMemcpy(matKernelPart, input + idx, kernelRows * kernelCols * sizeof(float), cudaMemcpyDeviceToDevice);
        CudaKernels::matrixMultiplicationWithFloatKernel();
    }
}