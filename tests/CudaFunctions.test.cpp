#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
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
