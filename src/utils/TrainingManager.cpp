#include "TrainingManager.hpp"

#include "Metrics/Metrics.hpp"
#include "Metrics/TrainingMetricsGui.hpp"
#include "MetricsCsvWriter.hpp"

#include <VNN.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

namespace TrainingManagerConfig
{
constexpr size_t TRAINING_STEPS = 30000;
constexpr size_t EVALUATION_INTERVAL = 150;
constexpr size_t GRADIENT_CHECK_SAMPLE_COUNT = 16;
}

namespace TrainingManagerCallbacks
{
struct TrainingState
{
    const TrainingManager *manager;
    Metrics::TrainingState *metricsState;
    TrainingMetricsGui *gui;
    NNG *network;
    NNG *evaluator;
    const float *testImages;
    const float *testLabels;
    size_t testCount;
};

bool trainingStepCallback(void *userData)
{
    TrainingState *state = static_cast<TrainingState *>(userData);
    const bool shouldContinue = Metrics::trainingStepCallback(state->metricsState);
    if (!shouldContinue &&
        state->metricsState->step == TrainingManagerConfig::TRAINING_STEPS)
    {
        const float accuracy = state->manager->test(
            *state->network,
            *state->evaluator,
            state->testImages,
            state->testLabels,
            state->testCount);
        state->gui->queueFinalTestAccuracy(accuracy);
    }

    return shouldContinue;
}
}

void TrainingManager::run(
    const float *trainingImages,
    std::vector<float> &trainingLabels,
    const float *testImages,
    const std::vector<float> &testLabels,
    size_t trainingCount,
    size_t testCount) const
{
    if (trainingImages == nullptr || testImages == nullptr ||
        trainingCount < TrainingManagerConfig::GRADIENT_CHECK_SAMPLE_COUNT ||
        trainingLabels.size() < trainingCount ||
        testCount == 0 || testLabels.size() < testCount)
    {
        throw std::invalid_argument("Ungueltige Daten fuer das Training");
    }

    NNG network(trainingCount);
    network.initilizeYMatrix(trainingLabels.data());
    network.setInputData(trainingImages);

    NNG evaluator(testCount);
    TrainingMetricsGui gui;
    gui.setImageData(trainingImages, trainingLabels.data(), trainingCount, 28, 28);
    MetricsCsvWriter metricsCsv("training_metrics.csv");
    gui.render();

    std::unique_ptr<float[]> gradientCheckImages(
        new float[SIZE_INPUT_LAYER * TrainingManagerConfig::GRADIENT_CHECK_SAMPLE_COUNT]);
    for (size_t feature = 0; feature < SIZE_INPUT_LAYER; ++feature)
    {
        for (size_t sample = 0;
             sample < TrainingManagerConfig::GRADIENT_CHECK_SAMPLE_COUNT;
             ++sample)
        {
            gradientCheckImages[
                feature * TrainingManagerConfig::GRADIENT_CHECK_SAMPLE_COUNT + sample] =
                trainingImages[feature * trainingCount + sample];
        }
    }

    Metrics::TrainingState trainingState{
        &network,
        &evaluator,
        &gui,
        &metricsCsv,
        testImages,
        testLabels.data(),
        trainingLabels.data(),
        TrainingManagerConfig::TRAINING_STEPS,
        TrainingManagerConfig::EVALUATION_INTERVAL,
        0,
        gradientCheckImages.get()};
    trainingState.activationBufferSizes = {
        SIZE_FIRST_LAYER * testCount,
        SIZE_SECOND_LAYER * testCount,
        SIZE_THIRD_LAYER * testCount};

    TrainingManagerCallbacks::TrainingState callbackState{
        this,
        &trainingState,
        &gui,
        &network,
        &evaluator,
        testImages,
        testLabels.data(),
        testCount};
    std::array<std::vector<float>, 3> hostActivationBuffers;
    for (size_t layer = 0; layer < hostActivationBuffers.size(); ++layer)
    {
        hostActivationBuffers[layer].resize(trainingState.activationBufferSizes[layer]);
        trainingState.hostActivationBuffers[layer] = hostActivationBuffers[layer].data();
    }

    gui.setTrainingStepCallback(
        TrainingManagerCallbacks::trainingStepCallback,
        &callbackState);
    gui.run();
}

float TrainingManager::test(
    NNG &network,
    NNG &evaluator,
    const float *testImages,
    const float *testLabels,
    size_t testCount) const
{
    if (testImages == nullptr || testLabels == nullptr || testCount == 0)
    {
        throw std::invalid_argument("Ungueltige Daten fuer den Test");
    }

    evaluator.copyParametersFrom(network);
    return evaluator.evaluate(testImages, testLabels).accuracy;
}
