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
    try
    {
        constexpr size_t trainingCount = 40000;
        constexpr size_t requestedTestCount = 10000;
        constexpr size_t TRAININGS_STEPS = 100;
        constexpr size_t EVALUATION_INTERVAL = 10;

        std::vector<float> labels = ImagePreProcessor::readLabelsAsFloat();
        if (labels.size() < trainingCount)
        {
            throw std::runtime_error("Weniger als 40000 Labels geladen");
        }
        std::vector<float> testLabels = ImagePreProcessor::readTestLabels();
        const size_t testCount = std::min(requestedTestCount, testLabels.size());
        if (testCount == 0)
            throw std::runtime_error("Keine Test-Labels geladen");

        std::unique_ptr<float[]> images(ImagePreProcessor::loadImageAsFLoat(trainingCount));
        std::unique_ptr<float[]> testImages(ImagePreProcessor::loadImageTestData(testCount));
        NNG network(trainingCount);
        network.initilizeYMatrix(labels.data());
        network.setInputData(images.get());

        NNG evaluator(testCount);
        TrainingMetricsGui gui;
        MetricsCsvWriter metricsCsv("training_metrics.csv");
        gui.render();

        constexpr size_t GRADIENT_CHECK_SAMPLE_COUNT = 16;
        Metrics::TrainingState trainingState{
            &network,
            &evaluator,
            &gui,
            &metricsCsv,
            testImages.get(),
            testLabels.data(),
            labels.data(),
            TRAININGS_STEPS,
            EVALUATION_INTERVAL};
        trainingState.gradientCheckImages.resize(SIZE_INPUT_LAYER * GRADIENT_CHECK_SAMPLE_COUNT);
        for (size_t feature = 0; feature < SIZE_INPUT_LAYER; ++feature)
        {
            for (size_t sample = 0; sample < GRADIENT_CHECK_SAMPLE_COUNT; ++sample)
            {
                trainingState.gradientCheckImages[feature * GRADIENT_CHECK_SAMPLE_COUNT + sample] =
                    images[feature * trainingCount + sample];
            }
        }
        gui.setTrainingStepCallback(Metrics::trainingStepCallback, &trainingState);
        gui.run();

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Programmfehler: " << error.what() << '\n';
        return 1;
    }
}