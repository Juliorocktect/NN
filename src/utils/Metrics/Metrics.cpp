#include "Metrics/Metrics.hpp"
#include "Metrics/TrainingMetricsGui.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace Metrics
{
ActivationDiagnostics calculateActivationDiagnostics(
    const std::array<float *, 3> &activations,
    const std::array<size_t, 3> &activationCounts)
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

bool trainingStepCallback(void *userData)
{
    TrainingState *state = static_cast<TrainingState *>(userData);
    ++state->step;
    const float trainingLoss = state->network->trainStep();
    if (state->step % state->evaluationInterval == 0 ||
        state->step == 1 ||
        state->step == state->trainingSteps)
    {
        state->evaluator->copyParametersFrom(*state->network);
        const EvaluationMetrics metrics =
            state->evaluator->evaluate(state->testImages, state->testLabels);
        std::array<GVector, 3> deviceActivations;
        state->evaluator->copyHiddenLayerActivations(deviceActivations);
        for (size_t layer = 0; layer < deviceActivations.size(); ++layer)
        {
            const size_t activationCount = deviceActivations[layer].getSize();
            if (activationCount != state->activationBufferSizes[layer])
                throw std::runtime_error("Unexpected hidden-layer activation buffer size");
            if (activationCount == 0)
                continue;

            const cudaError_t error = cudaMemcpy(
                state->hostActivationBuffers[layer],
                deviceActivations[layer].getVector(),
                activationCount * sizeof(float),
                cudaMemcpyDeviceToHost);
            if (error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
        }
        const ActivationDiagnostics activationDiagnostics =
            calculateActivationDiagnostics(
                state->hostActivationBuffers,
                state->activationBufferSizes);
        state->activationHistory.push_back({state->step, activationDiagnostics});

        TrainingMetricsSnapshot snapshot{
            state->step,
            trainingLoss,
            metrics.loss,
            metrics.accuracy,
            metrics.actualCounts,
            metrics.predictedCounts,
            metrics.confusionCounts,
            metrics.normalizedConfusion,
            metrics.classRecall,
            metrics.meanConfidence,
            activationDiagnostics};
        state->metricsCsv->writeMetrics(
            state->step, trainingLoss, metrics, activationDiagnostics);
        state->gui->queueMetrics(snapshot);
        std::cout << "Schritt " << state->step
                  << " | Trainings-Loss: " << trainingLoss
                  << " | Test-Loss: " << metrics.loss
                  << " | Testgenauigkeit: " << metrics.accuracy
                  << " | Vorhersagen je Klasse:";
        for (size_t category = 0; category < metrics.predictedCounts.size(); ++category)
            std::cout << ' ' << category << '=' << metrics.predictedCounts[category];
        std::cout << '\n';
    }

    if (state->step == state->trainingSteps)
    {
        constexpr size_t GRADIENT_CHECK_SAMPLE_COUNT = 16;
        const bool passed = state->network->runGradientCheck(
            state->gradientCheckImages,
            state->trainingLabels,
            GRADIENT_CHECK_SAMPLE_COUNT);
        if (!passed)
            std::cerr << "Warnung: Gradiententest am Trainingsende ist fehlgeschlagen.\n";
        std::cout.flush();
        std::cerr.flush();
        return false;
    }

    return true;
}
}
