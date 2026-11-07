#include "TrainingDataLoader.hpp"

#include "ImageLoading.h"

#include <algorithm>
#include <stdexcept>

TrainingDataset TrainingDataLoader::load(const TrainingConfig &config) const
{
    if (config.trainingCount == 0 || config.testCount == 0)
    {
        throw std::invalid_argument("Training and test counts must be positive");
    }

    TrainingDataset dataset;
    dataset.trainingLabels = ImagePreProcessor::readLabelsAsFloat();
    if (dataset.trainingLabels.size() < config.trainingCount)
    {
        throw std::runtime_error("Nicht genuegend Trainingslabels geladen");
    }

    dataset.testLabels = ImagePreProcessor::readTestLabels();
    dataset.testCount = std::min(config.testCount, dataset.testLabels.size());
    if (dataset.testCount == 0)
    {
        throw std::runtime_error("Keine Testlabels geladen");
    }

    dataset.trainingCount = config.trainingCount;
    dataset.trainingImages.reset(
        ImagePreProcessor::loadImageAsFLoat(dataset.trainingCount));
    dataset.testImages.reset(
        ImagePreProcessor::loadImageTestData(dataset.testCount));
    return dataset;
}
