#include <ImageLoading.h>
#include "TrainingManager.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int main()
{
    constexpr size_t TRAINING_COUNT = 5000;
    constexpr size_t REQUESTED_TEST_COUNT = 10000;

    try
    {
        std::vector<float> labels = ImagePreProcessor::readLabelsAsFloat();
        if (labels.size() < TRAINING_COUNT)
        {
            throw std::runtime_error("Weniger als 5000 Labels geladen");
        }

        std::vector<float> testLabels = ImagePreProcessor::readTestLabels();
        const size_t testCount = std::min(REQUESTED_TEST_COUNT, testLabels.size());
        if (testCount == 0)
        {
            throw std::runtime_error("Keine Test-Labels geladen");
        }

        std::unique_ptr<float[]> images(
            ImagePreProcessor::loadImageAsFLoat(TRAINING_COUNT));
        std::unique_ptr<float[]> testImages(
            ImagePreProcessor::loadImageTestData(testCount));

        TrainingManager trainingManager;
        trainingManager.run(
            images.get(),
            labels,
            testImages.get(),
            testLabels,
            TRAINING_COUNT,
            testCount);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Programmfehler: " << error.what() << '\n';
        return 1;
    }
}
