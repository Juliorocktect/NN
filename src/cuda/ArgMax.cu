#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

double *executeArgmaxKernel(const double *mat, int rows, int cols)
{
    double *d_mat;
    double *h_mat_res = new double[cols];
    double *d_vec_res;
    size_t sizeVecRes = cols * sizeof(double);
    size_t sizeMat = rows * cols * sizeof(double);
    cudaMalloc((void **)&d_mat, sizeMat);
    cudaMalloc((void **)&d_vec_res, sizeVecRes);

    cudaMemcpy(d_mat, mat, sizeMat, cudaMemcpyHostToDevice);
    int threadsPerBlock = 1;
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;
    argmaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_mat, d_vec_res, rows, cols);
    cudaMemcpy(h_mat_res, d_vec_res, sizeVecRes, cudaMemcpyDeviceToHost);
    cudaFree(d_mat);
    cudaFree(d_vec_res);
    return h_mat_res;
}
__global__ void argmaxKernel(const double *mat, double *mat_result, int rows, int cols)
{
    int col = blockIdx.x;

    float maxVal = mat[col * rows];
    int index = 0;
    for (int i = 1; i < rows; i++)
    {

        float v = mat[col * rows + i];
        if (v > maxVal)
            maxVal = v;
        index = col * rows + i;
    }

    mat_result[col] = index;
}

__global__ void CudaKernels::argmax(const float *mat, int rows, int cols, float *result)
{
    int col = blockIdx.x;

    float maxVal = mat[col * rows];
    int index = 0;
    for (int i = 1; i < rows; i++)
    {

        float v = mat[col * rows + i];
        if (v > maxVal)
        {
            maxVal = v;
            index = i;
        }
    }
    result[col] = index;
}

void CudaLaunchers::argmax(const float *mat, int rows, int cols, float *result)
{
    int threadsPerBlock = 1;
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;
    CudaKernels::argmax<<<threadsPerBlock, blocksPerGrid>>>(mat, rows, cols, result);
    cudaDeviceSynchronize();
}