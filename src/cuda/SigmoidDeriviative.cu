#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__device__ double sigmoidDeriviative(double x)
{
    double y = sigmoid(x);
    return y * (1 - y);
}
__device__ float CudaKernels::sigmoidDeriviativeFunction(float x)
{
    float y = sigmoid(x);
    return y * (1 - y);
}

__global__ void applySigmoidDeriviative(const double *mat1, double *mat_result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        mat_result[idx] = sigmoidDeriviative(mat1[idx]);
    }
}
double *executeSigmoidDeriviativeKernel(const double *mat1, int rows, int cols)
{
    double *d_mat_result;
    double *d_mat1;
    size_t n = rows * cols * sizeof(double);
    int o = rows * cols;
    double *h_result = new double[o];
    cudaMalloc((void **)&d_mat_result, n);
    cudaMalloc((void **)&d_mat1, n);
    cudaMemcpy(d_mat1, mat1, n, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (o + threadsPerBlock - 1) / threadsPerBlock;
    applySigmoidDeriviative<<<threadsPerBlock, blocksPerGrid>>>(d_mat1, d_mat_result, o);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_mat1, n, cudaMemcpyDeviceToHost);
    cudaFree(&d_mat_result);
    cudaFree(&d_mat1);
    return h_result;
}

float *CudaLaunchers::sigmoidDeriviative(float *input, float *output, size_t size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    CudaKernels::sigmoidDeriviative<<<blocksPerGrid, threadsPerBlock>>>(input, output, size);
    cudaDeviceSynchronize();
    return output;
}

__global__ void CudaKernels::sigmoidDeriviative(float *input, float *output, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = CudaKernels::sigmoidDeriviativeFunction(input[idx]);
    }
}