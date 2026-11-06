#include <VNN.h>
#include <ImageLoading.h>
#include "Metrics/Metrics.hpp"
#include "MetricsCsvWriter.hpp"
#include "TrainingMetricsGui.hpp"
#include <memory>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>

int main()
{
    float *images = nullptr;
    float *testImages = nullptr;
    float *gradientCheckImages = nullptr;
    std::array<float *, 3> hostActivationBuffers{};
    try
    {
        constexpr size_t trainingCount = 40000;
        constexpr size_t requestedTestCount = 10000;
        constexpr size_t TRAININGS_STEPS = 1000;
        constexpr size_t EVALUATION_INTERVAL = 50;

        std::vector<float> labels = ImagePreProcessor::readLabelsAsFloat();
        if (labels.size() < trainingCount)
        {
            throw std::runtime_error("Weniger als 40000 Labels geladen");
        }
        std::vector<float> testLabels = ImagePreProcessor::readTestLabels();
        const size_t testCount = std::min(requestedTestCount, testLabels.size());
        if (testCount == 0)
            throw std::runtime_error("Keine Test-Labels geladen");

        images = ImagePreProcessor::loadImageAsFLoat(trainingCount);
        testImages = ImagePreProcessor::loadImageTestData(testCount);

        NNG network(trainingCount);
        network.initilizeYMatrix(labels.data());
        network.setInputData(images);

        NNG evaluator(testCount);
        TrainingMetricsGui gui;
        gui.setImageData(images, labels.data(), trainingCount, 28, 28);
        MetricsCsvWriter metricsCsv("training_metrics.csv");
        gui.render();

        constexpr size_t GRADIENT_CHECK_SAMPLE_COUNT = 16;
        gradientCheckImages = new float[SIZE_INPUT_LAYER * GRADIENT_CHECK_SAMPLE_COUNT];
        Metrics::TrainingState trainingState{
            &network,
            &evaluator,
            &gui,
            &metricsCsv,
            testImages,
            testLabels.data(),
            labels.data(),
            TRAININGS_STEPS,
            EVALUATION_INTERVAL,
            0,
            gradientCheckImages};
        trainingState.activationBufferSizes = {
            SIZE_FIRST_LAYER * testCount,
            SIZE_SECOND_LAYER * testCount,
            SIZE_THIRD_LAYER * testCount};
        for (size_t feature = 0; feature < SIZE_INPUT_LAYER; ++feature)
        {
            for (size_t sample = 0; sample < GRADIENT_CHECK_SAMPLE_COUNT; ++sample)
            {
                gradientCheckImages[feature * GRADIENT_CHECK_SAMPLE_COUNT + sample] =
                    images[feature * trainingCount + sample];
            }
        }
        for (size_t layer = 0; layer < hostActivationBuffers.size(); ++layer)
        {
            hostActivationBuffers[layer] = new float[trainingState.activationBufferSizes[layer]];
            trainingState.hostActivationBuffers[layer] = hostActivationBuffers[layer];
        }
        gui.setTrainingStepCallback(Metrics::trainingStepCallback, &trainingState);
        gui.run();

        for (float *&activationBuffer : hostActivationBuffers)
        {
            delete[] activationBuffer;
            activationBuffer = nullptr;
        }
        delete[] gradientCheckImages;
        gradientCheckImages = nullptr;
        delete[] testImages;
        testImages = nullptr;
        delete[] images;
        images = nullptr;
        return 0;
    }
    catch (const std::exception &error)
    {
        for (float *activationBuffer : hostActivationBuffers)
            delete[] activationBuffer;
        delete[] gradientCheckImages;
        delete[] testImages;
        delete[] images;
        std::cerr << "Programmfehler: " << error.what() << '\n';
        return 1;
    }
}