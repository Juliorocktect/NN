#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <numeric>
#include <vector>
#include "CudaLaunchers.cuh"

TEST(CudaCrossEntropyLossTest, ComputesMeanLossForSingleSample)
{
    const int numSamples = 1;
    std::vector<float> y_hat(10, 0.0f);
    y_hat[1] = 0.9f;
    std::vector<float> labels = {1.0f};

    float loss = CudaLaunchers::sumCrossEntropyLoss(y_hat.data(), labels.data(), numSamples);
    float expected = -std::log(0.9f);

    EXPECT_NEAR(loss, expected, 1e-5f);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(CudaCrossEntropyLossTest, ComputesMeanLossForMultipleSamples)
{
    const int numSamples = 2;
    std::vector<float> y_hat(10 * numSamples, 0.0f);
    y_hat[0 * 10 + 0] = 0.9f; // sample 0, class 0
    y_hat[1 * 10 + 2] = 0.8f; // sample 1, class 2
    std::vector<float> labels = {0.0f, 2.0f};

    float loss = CudaLaunchers::sumCrossEntropyLoss(y_hat.data(), labels.data(), numSamples);
    float expected = (-std::log(0.9f) - std::log(0.8f)) / 2.0f;

    EXPECT_NEAR(loss, expected, 1e-5f);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(CudaCrossEntropyLossTest, ReturnsZeroForInvalidLabel)
{
    const int numSamples = 2;
    std::vector<float> y_hat(10 * numSamples, 0.0f);
    y_hat[0 * 10 + 0] = 0.7f; // sample 0, class 0
    y_hat[1 * 10 + 1] = 0.6f; // sample 1, class 1
    std::vector<float> labels = {0.0f, 10.0f};

    float loss = CudaLaunchers::sumCrossEntropyLoss(y_hat.data(), labels.data(), numSamples);
    float expected = (-std::log(0.7f) + 0.0f) / 2.0f;

    EXPECT_NEAR(loss, expected, 1e-5f);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(CudaLaunchersHotEncodeYMatrixTest, EncodesRepeatedLabelsCorrectly)
{
    const size_t numSamples = 4;
    std::vector<float> labels_h = {3.0f, 3.0f, 3.0f, 3.0f};
    std::vector<float> expected(numSamples * 10, 0.0f);
    for (size_t i = 0; i < numSamples; ++i)
    {
        expected[i * 10 + 3] = 1.0f;
    }

    float *labels_d = nullptr;
    float *result_d = nullptr;
    cudaMalloc(&labels_d, numSamples * sizeof(float));
    cudaMalloc(&result_d, numSamples * 10 * sizeof(float));
    cudaMemset(result_d, 0, numSamples * 10 * sizeof(float));
    cudaMemcpy(labels_d, labels_h.data(), numSamples * sizeof(float), cudaMemcpyHostToDevice);

    CudaLaunchers::hotEncodeYMatrix(labels_d, result_d, numSamples);

    std::vector<float> result_h(numSamples * 10, 0.0f);
    cudaMemcpy(result_h.data(), result_d, numSamples * 10 * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(result_h, expected);

    cudaFree(labels_d);
    cudaFree(result_d);
}

TEST(CudaLaunchersHotEncodeYMatrixTest, EncodesEveryClassCorrectly)
{
    const size_t numSamples = 10;
    std::vector<float> labels_h(numSamples);
    std::iota(labels_h.begin(), labels_h.end(), 0.0f);

    std::vector<float> expected(numSamples * 10, 0.0f);
    for (size_t i = 0; i < numSamples; ++i)
    {
        expected[i * 10 + static_cast<int>(labels_h[i])] = 1.0f;
    }

    float *labels_d = nullptr;
    float *result_d = nullptr;
    cudaMalloc(&labels_d, numSamples * sizeof(float));
    cudaMalloc(&result_d, numSamples * 10 * sizeof(float));
    cudaMemset(result_d, 0, numSamples * 10 * sizeof(float));
    cudaMemcpy(labels_d, labels_h.data(), numSamples * sizeof(float), cudaMemcpyHostToDevice);

    CudaLaunchers::hotEncodeYMatrix(labels_d, result_d, numSamples);

    std::vector<float> result_h(numSamples * 10, 0.0f);
    cudaMemcpy(result_h.data(), result_d, numSamples * 10 * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(result_h, expected);

    cudaFree(labels_d);
    cudaFree(result_d);
}
