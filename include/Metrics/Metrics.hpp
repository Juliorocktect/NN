#ifndef METRICS_METRICS_HPP
#define METRICS_METRICS_HPP

#include "Metrics/EvaluationMetrics.hpp"

#include <array>
#include <cstddef>

namespace Metrics
{
inline constexpr size_t ACTIVATION_DERIVATIVE_BIN_COUNT = 5;

/** @brief Saturation and sigmoid-derivative statistics for one hidden layer. */
struct ActivationLayerMetrics
{
    /** @brief Number of activation values included in the statistics. */
    size_t activationCount = 0;
    /** @brief Number of activation values outside the unsaturated interval [0.01, 0.99]. */
    size_t saturatedCount = 0;
    /** @brief Fraction of activation values counted as saturated. */
    double saturatedFraction = 0.0;
    /** @brief Mean sigmoid derivative across all activation values. */
    double meanSigmoidDerivative = 0.0;
    /** @brief Number of derivatives in each configured derivative range. */
    std::array<size_t, ACTIVATION_DERIVATIVE_BIN_COUNT> derivativeBinCounts{};
    /** @brief Fraction of derivatives in each configured derivative range. */
    std::array<double, ACTIVATION_DERIVATIVE_BIN_COUNT> derivativeBinFractions{};
};

/** @brief Activation diagnostics for all hidden layers. */
struct ActivationDiagnostics
{
    /** @brief Per-layer diagnostics in network order. */
    std::array<ActivationLayerMetrics, 3> layers{};
};

/** @brief Activation diagnostics recorded at a particular training step. */
struct ActivationHistoryEntry
{
    /** @brief Training step at which the diagnostics were collected. */
    size_t step = 0;
    /** @brief Collected activation statistics. */
    ActivationDiagnostics diagnostics;
};

/** @brief Immutable values published to the GUI and CSV writer for a training step. */
struct TrainingMetricsSnapshot
{
    /** @brief Training step represented by this snapshot. */
    size_t step;
    /** @brief Loss reported by the training step. */
    float trainingLoss;
    /** @brief Evaluation metrics computed on the test dataset. */
    EvaluationMetrics evaluationMetrics;
    /** @brief Hidden-layer activation diagnostics for this step. */
    ActivationDiagnostics activationDiagnostics;
};

/** @brief Calculates descriptive statistics for sigmoid activations in each hidden layer. */
class ActivationDiagnosticsCalculator
{
public:
    /**
     * @brief Calculates saturation and derivative distributions for hidden-layer activations.
     *
     * @param activations Host buffers containing activation values for each hidden layer.
     * @param activationCounts Number of values in each corresponding buffer.
     * @return Per-layer activation diagnostics.
     * @throws std::invalid_argument If a buffer is null, empty, or contains an invalid activation.
     */
    ActivationDiagnostics calculate(
        const std::array<float *, 3> &activations,
        const std::array<size_t, 3> &activationCounts) const;
};
}

#endif
