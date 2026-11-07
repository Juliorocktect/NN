#include "TrainingController.hpp"

#include "EvaluationService.hpp"
#include "Metrics/Metrics.hpp"
#include "Metrics/TrainingMetricsGui.hpp"
#include "MetricsCsvWriter.hpp"

#include <VNN.h>

#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

struct TrainingController::TrainingState
{
    NNG *network;
    NNG *evaluator;
    const TrainingDataset *dataset;
    const TrainingConfig *config;
    TrainingMetricsGui *gui;
    MetricsCsvWriter *metricsCsv;
    EvaluationService evaluationService;
    size_t step = 0;
    std::unique_ptr<float[]> gradientCheckImages;
    std::array<std::vector<float>, 3> hostActivationBuffers;
    std::array<size_t, 3> activationBufferSizes{};
};

namespace
{
void copyGradientCheckSamples(
    float *gradientCheckImages,
    const float *trainingImages,
    size_t trainingCount,
    size_t sampleCount)
{
    for (size_t feature = 0; feature < SIZE_INPUT_LAYER; ++feature)
    {
        for (size_t sample = 0; sample < sampleCount; ++sample)
        {
            gradientCheckImages[feature * sampleCount + sample] =
                trainingImages[feature * trainingCount + sample];
        }
    }
}

Metrics::ActivationDiagnostics calculateEvaluationActivationDiagnostics(
    NNG &evaluator,
    std::array<std::vector<float>, 3> &hostActivationBuffers,
    const std::array<size_t, 3> &activationBufferSizes)
{
    std::array<GVector, 3> deviceActivations;
    evaluator.copyHiddenLayerActivations(deviceActivations);
    std::array<float *, 3> hostActivationData{};

    for (size_t layer = 0; layer < deviceActivations.size(); ++layer)
    {
        const size_t activationCount = deviceActivations[layer].getSize();
        if (activationCount != activationBufferSizes[layer])
        {
            throw std::runtime_error("Unexpected hidden-layer activation buffer size");
        }

        if (activationCount == 0)
        {
            continue;
        }

        std::vector<float> &hostBuffer = hostActivationBuffers[layer];
        const cudaError_t error = cudaMemcpy(
            hostBuffer.data(),
            deviceActivations[layer].getVector(),
            activationCount * sizeof(float),
            cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(error));
        }
        hostActivationData[layer] = hostBuffer.data();
    }

    const Metrics::ActivationDiagnosticsCalculator calculator;
    return calculator.calculate(
        hostActivationData,
        activationBufferSizes);
}
}

bool TrainingController::runTrainingStep(void *userData)
{
    TrainingState *state = static_cast<TrainingState *>(userData);
    ++state->step;
    const float trainingLoss = state->network->trainStep();
    const bool evaluationDue =
        state->step % state->config->evaluationInterval == 0 ||
        state->step == 1 ||
        state->step == state->config->trainingSteps;

    if (evaluationDue)
    {
        state->evaluator->copyParametersFrom(*state->network);
        const EvaluationMetrics evaluationMetrics = state->evaluationService.evaluate(
            *state->evaluator,
            state->dataset->testImages.get(),
            state->dataset->testLabels.data(),
            state->dataset->testCount);
        const Metrics::ActivationDiagnostics activationDiagnostics =
            calculateEvaluationActivationDiagnostics(
                *state->evaluator,
                state->hostActivationBuffers,
                state->activationBufferSizes);

        const Metrics::TrainingMetricsSnapshot snapshot{
            state->step,
            trainingLoss,
            evaluationMetrics,
            activationDiagnostics};
        state->metricsCsv->writeMetrics(
            state->step,
            trainingLoss,
            evaluationMetrics,
            activationDiagnostics);
        state->gui->queueMetrics(snapshot);

        std::cout << "Schritt " << state->step
                  << " | Trainings-Loss: " << trainingLoss
                  << " | Test-Loss: " << evaluationMetrics.loss
                  << " | Testgenauigkeit: " << evaluationMetrics.accuracy
                  << " | Vorhersagen je Klasse:";
        for (size_t category = 0; category < evaluationMetrics.predictedCounts.size(); ++category)
        {
            std::cout << ' ' << category << '=' << evaluationMetrics.predictedCounts[category];
        }
        std::cout << '\n';

        if (state->step == state->config->trainingSteps)
        {
            state->gui->queueFinalTestAccuracy(evaluationMetrics.accuracy);
        }
    }

    if (state->step == state->config->trainingSteps)
    {
        const bool passed = state->network->runGradientCheck(
            state->gradientCheckImages.get(),
            state->dataset->trainingLabels.data(),
            state->config->gradientCheckSampleCount);
        if (!passed)
        {
            std::cerr << "Warnung: Gradiententest am Trainingsende ist fehlgeschlagen.\n";
        }
        std::cout.flush();
        std::cerr.flush();
        return false;
    }

    return true;
}

void TrainingController::run(
    TrainingDataset &dataset,
    const TrainingConfig &config) const
{
    const bool imageDimensionsOverflow =
        config.imageHeight != 0 &&
        config.imageWidth > std::numeric_limits<size_t>::max() / config.imageHeight;
    if (dataset.trainingImages == nullptr || dataset.testImages == nullptr ||
        config.imageWidth == 0 || config.imageHeight == 0 ||
        imageDimensionsOverflow ||
        config.imageWidth * config.imageHeight != SIZE_INPUT_LAYER ||
        dataset.trainingCount < config.gradientCheckSampleCount ||
        dataset.trainingLabels.size() < dataset.trainingCount ||
        dataset.testCount == 0 || dataset.testLabels.size() < dataset.testCount ||
        config.trainingSteps == 0 || config.evaluationInterval == 0 ||
        config.gradientCheckSampleCount == 0 ||
        config.gradientCheckSampleCount > dataset.trainingCount)
    {
        throw std::invalid_argument("Ungueltige Trainingsdaten oder Trainingskonfiguration");
    }

    NNG network(dataset.trainingCount);
    network.initilizeYMatrix(dataset.trainingLabels.data());
    network.setInputData(dataset.trainingImages.get());

    NNG evaluator(dataset.testCount);
    TrainingMetricsGui gui;
    gui.setImageData(
        dataset.trainingImages.get(),
        dataset.trainingLabels.data(),
        dataset.trainingCount,
        config.imageWidth,
        config.imageHeight);
    MetricsCsvWriter metricsCsv("training_metrics.csv");
    gui.render();

    TrainingState state{
        &network,
        &evaluator,
        &dataset,
        &config,
        &gui,
        &metricsCsv};
    state.gradientCheckImages.reset(
        new float[SIZE_INPUT_LAYER * config.gradientCheckSampleCount]);
    copyGradientCheckSamples(
        state.gradientCheckImages.get(),
        dataset.trainingImages.get(),
        dataset.trainingCount,
        config.gradientCheckSampleCount);
    state.activationBufferSizes = {
        SIZE_FIRST_LAYER * dataset.testCount,
        SIZE_SECOND_LAYER * dataset.testCount,
        SIZE_THIRD_LAYER * dataset.testCount};

    for (size_t layer = 0; layer < state.hostActivationBuffers.size(); ++layer)
    {
        state.hostActivationBuffers[layer].resize(state.activationBufferSizes[layer]);
    }

    gui.setTrainingStepCallback(TrainingController::runTrainingStep, &state);
    gui.run();
}
