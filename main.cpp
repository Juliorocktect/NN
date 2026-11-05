#include <VNN.h>
#include <ImageLoading.h>
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
        constexpr size_t TRAININGS_STEPS = 20000;
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

        size_t step = 0;
        gui.setTrainingStepCallback([&]()
                                    {
            ++step;
            const float trainingLoss = network.trainStep();
            if (step % EVALUATION_INTERVAL == 0 || step == 1 || step == TRAININGS_STEPS)
            {
                evaluator.copyParametersFrom(network);
                const EvaluationMetrics metrics = evaluator.evaluate(testImages.get(), testLabels.data());
                metricsCsv.writeMetrics(step, trainingLoss, metrics);
                gui.addMetrics(step, trainingLoss, metrics.loss, metrics.accuracy);
                gui.setDiagnostics(
                    metrics.actualCounts,
                    metrics.predictedCounts,
                    metrics.confusionCounts,
                    metrics.normalizedConfusion,
                    metrics.classRecall,
                    metrics.meanConfidence);
                gui.render();
                std::cout << "Schritt " << step
                          << " | Trainings-Loss: " << trainingLoss
                          << " | Test-Loss: " << metrics.loss
                          << " | Testgenauigkeit: " << metrics.accuracy
                          << " | Vorhersagen je Klasse:";
                for (size_t category = 0; category < metrics.predictedCounts.size(); ++category)
                    std::cout << ' ' << category << '=' << metrics.predictedCounts[category];
                std::cout << '\n';
            }
            return step < TRAININGS_STEPS; });
        gui.run();

            constexpr size_t gradientCheckSampleCount = 16;
            std::vector<float> gradientCheckImages(SIZE_INPUT_LAYER * gradientCheckSampleCount);
            for (size_t feature = 0; feature < SIZE_INPUT_LAYER; ++feature)
            {
                for (size_t sample = 0; sample < gradientCheckSampleCount; ++sample)
                {
                    gradientCheckImages[feature * gradientCheckSampleCount + sample] =
                        images[feature * trainingCount + sample];
                }
            }
            if (!network.runGradientCheck(
                    gradientCheckImages.data(),
                    labels.data(),
                    gradientCheckSampleCount))
            {
                std::cerr << "Warnung: Gradiententest am Trainingsende ist fehlgeschlagen.\n";
            }

            return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Programmfehler: " << error.what() << '\n';
        return 1;
    }
}