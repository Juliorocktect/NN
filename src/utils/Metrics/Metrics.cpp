#include "Metrics/Metrics.hpp"

#include <cmath>
#include <stdexcept>

namespace Metrics
{
ActivationDiagnostics ActivationDiagnosticsCalculator::calculate(
    const std::array<float *, 3> &activations,
    const std::array<size_t, 3> &activationCounts) const
{
    ActivationDiagnostics diagnostics;
    for (size_t layerIndex = 0; layerIndex < activations.size(); ++layerIndex)
    {
        if (activations[layerIndex] == nullptr || activationCounts[layerIndex] == 0)
            throw std::invalid_argument("Activation diagnostics require non-empty layers");

        ActivationLayerMetrics &layerMetrics = diagnostics.layers[layerIndex];
        layerMetrics.activationCount = activationCounts[layerIndex];
        double derivativeSum = 0.0;
        for (size_t index = 0; index < activationCounts[layerIndex]; ++index)
        {
            const float activation = activations[layerIndex][index];
            if (!std::isfinite(activation) || activation < 0.0f || activation > 1.0f)
                throw std::invalid_argument("Sigmoid activations must be finite values from 0 to 1");

            if (activation < 0.01f || activation > 0.99f)
                ++layerMetrics.saturatedCount;

            const double derivative =
                static_cast<double>(activation) * (1.0 - static_cast<double>(activation));
            derivativeSum += derivative;
            size_t bin = 4;
            if (derivative < 0.01)
                bin = 0;
            else if (derivative < 0.05)
                bin = 1;
            else if (derivative < 0.10)
                bin = 2;
            else if (derivative < 0.20)
                bin = 3;
            ++layerMetrics.derivativeBinCounts[bin];
        }

        const double count = static_cast<double>(layerMetrics.activationCount);
        layerMetrics.saturatedFraction =
            static_cast<double>(layerMetrics.saturatedCount) / count;
        layerMetrics.meanSigmoidDerivative = derivativeSum / count;
        for (size_t bin = 0; bin < layerMetrics.derivativeBinCounts.size(); ++bin)
        {
            layerMetrics.derivativeBinFractions[bin] =
                static_cast<double>(layerMetrics.derivativeBinCounts[bin]) / count;
        }
    }
    return diagnostics;
}
}
