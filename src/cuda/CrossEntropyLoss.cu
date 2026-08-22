#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

__global__ void crossEntropyLossKernel(const double *y_hat, const double *labels, double *loss, int numSamples)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples)
    {
        int label = static_cast<int>(labels[idx]);
        if (label >= 0 && label < 10)
        {
            double val = y_hat[label + idx * 10]; // Spaltenweise: label + idx*10
            if (val <= 0.0)
                val = 1e-8; // Schutz vor log(0)
            loss[idx] = -log(val);
        }
        else
        {
            loss[idx] = 0.0;
        }
    }
}
double executeCrossEntropyLoss(const double *y_hat, const double *labels, int numSamples)
{
    double *d_y_hat;
    double *d_labels;
    double *d_loss;
    size_t sizeYHAT = 10 * numSamples * sizeof(double);
    double *h_loss = new double[numSamples];

    cudaMalloc((void **)&d_y_hat, sizeYHAT);
    cudaMalloc((void **)&d_labels, numSamples * sizeof(double));
    cudaMalloc((void **)&d_loss, numSamples * sizeof(double));
    cudaMemcpy(d_y_hat, y_hat, sizeYHAT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, numSamples * sizeof(double), cudaMemcpyHostToDevice);
    int threadsPerBlock = 512;
    int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;
    crossEntropyLossKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y_hat, d_labels, d_loss, numSamples);
    cudaMemcpy(h_loss, d_loss, numSamples * sizeof(double), cudaMemcpyDeviceToHost);
    double sum = 0.0;
    for (int i = 0; i < numSamples; ++i)
        sum += h_loss[i];
    double meanLoss = sum / numSamples;
    delete[] h_loss;
    return meanLoss;
}
__global__ void CudaKernels::crossEntropyLossKernel(const float *y_hat, const float *labels, float *loss, int numSamples)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples)
    {
        int label = static_cast<int>(labels[idx]);
        if (label >= 0 && label < 10)
        {
            float val = y_hat[label + idx * 10];
            if (val <= 0.0)
            {
                val = 1e-8f;
            }
            loss[idx] = -logf(val);
        }
        else
        {
            loss[idx] = 0.0;
        }
    }
}

float CudaLaunchers::sumCrossEntropyLoss(const float *y_hat, const float *labels, int numSamples)
{
    if (y_hat == nullptr || labels == nullptr || numSamples <= 0)
        return 0.0f;

    int threadsPerBlock = 512;
    int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;
    float *h_loss = new float[numSamples];
    float *d_loss;
    cudaMalloc((void **)&d_loss, numSamples * sizeof(float));

    CudaKernels::crossEntropyLossKernel<<<blocksPerGrid, threadsPerBlock>>>(y_hat, labels, d_loss, numSamples);
    cudaDeviceSynchronize();
    cudaMemcpy(h_loss, d_loss, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0.0;
    for (int i = 0; i < numSamples; ++i)
        sum += h_loss[i];
    float meanLoss = sum / numSamples;
    delete[] h_loss;
    cudaFree(d_loss);
    return meanLoss;
}
