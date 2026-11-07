#ifndef TRAINING_DATASET_HPP
#define TRAINING_DATASET_HPP

#include <cstddef>
#include <memory>
#include <vector>

struct TrainingDataset
{
    std::vector<float> trainingLabels;
    std::vector<float> testLabels;
    std::unique_ptr<float[]> trainingImages;
    std::unique_ptr<float[]> testImages;
    size_t trainingCount = 0;
    size_t testCount = 0;
};

#endif
