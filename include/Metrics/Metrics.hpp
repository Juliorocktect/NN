#ifndef METRICS_METRICS_HPP
#define METRICS_METRICS_HPP

#include "Metrics/EvaluationMetrics.hpp"

#include <array>
#include <cstddef>

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
    EvaluationMetrics evaluationMetrics;
    ActivationDiagnostics activationDiagnostics;
};

class ActivationDiagnosticsCalculator
{
public:
    ActivationDiagnostics calculate(
        const std::array<float *, 3> &activations,
        const std::array<size_t, 3> &activationCounts) const;
};
}

#endif
