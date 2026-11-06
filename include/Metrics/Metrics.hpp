#ifndef METRICS_METRICS_HPP
#define METRICS_METRICS_HPP

#include "MetricsCsvWriter.hpp"
#include "VNN.h"

#include <array>
#include <cstddef>
#include <vector>

class TrainingMetricsGui;

namespace Metrics
{
inline constexpr size_t ACTIVATION_DERIVATIVE_BIN_COUNT = 5;

struct ActivationLayerMetrics
{
    size_t activationCount = 0;
    size_t saturatedCount = 0;
    double saturatedFraction = 0.0;
    double meanSigmoidDerivative = 0.0;
    std::array<size_t, ACTIVATION_DERIVATIVE_BIN_COUNT> derivativeBinCounts{};
    std::array<double, ACTIVATION_DERIVATIVE_BIN_COUNT> derivativeBinFractions{};
};

struct ActivationDiagnostics
{
    std::array<ActivationLayerMetrics, 3> layers{};
};

struct ActivationHistoryEntry
{
    size_t step = 0;
    ActivationDiagnostics diagnostics;
};

struct TrainingMetricsSnapshot
{
    size_t step;
    float trainingLoss;
    float testLoss;
    float testAccuracy;
    std::array<size_t, 10> actualCounts;
    std::array<size_t, 10> predictedCounts;
    std::array<size_t, 100> confusionCounts;
    std::array<double, 100> normalizedConfusion;
    std::array<float, 10> classRecall;
    float meanConfidence;
    ActivationDiagnostics activationDiagnostics;
};

struct TrainingState
{
    NNG *network;
    NNG *evaluator;
    TrainingMetricsGui *gui;
    MetricsCsvWriter *metricsCsv;
    const float *testImages;
    const float *testLabels;
    const float *trainingLabels;
    size_t trainingSteps;
    size_t evaluationInterval;
    size_t step = 0;
    float *gradientCheckImages = nullptr;
    std::array<float *, 3> hostActivationBuffers{};
    std::array<size_t, 3> activationBufferSizes{};
    std::vector<ActivationHistoryEntry> activationHistory;
};

ActivationDiagnostics calculateActivationDiagnostics(
    const std::array<float *, 3> &activations,
    const std::array<size_t, 3> &activationCounts);
bool trainingStepCallback(void *userData);
}

#endif
