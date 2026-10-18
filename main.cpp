#include <VNN.h>
#include <ImageLoading.h>
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
        constexpr size_t trainingSteps = 20000;
        constexpr size_t evaluationInterval = 100;

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
        gui.render();

        for (size_t step = 1; step <= trainingSteps && gui.processEvents(); ++step)
        {
            const float trainingLoss = network.trainStep();
            if (step % evaluationInterval == 0 || step == 1 || step == trainingSteps)
            {
                evaluator.copyParametersFrom(network);
                const auto [testLoss, testAccuracy] = evaluator.evaluate(testImages.get(), testLabels.data());
                gui.addMetrics(step, trainingLoss, testLoss, testAccuracy);
                gui.render();
                std::cout << "Schritt " << step
                          << " | Trainings-Loss: " << trainingLoss
                          << " | Test-Loss: " << testLoss
                          << " | Testgenauigkeit: " << testAccuracy << '\n';
            }
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Programmfehler: " << error.what() << '\n';
        return 1;
    }
}