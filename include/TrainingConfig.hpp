#ifndef TRAINING_CONFIG_HPP
#define TRAINING_CONFIG_HPP

#include <cstddef>

struct TrainingConfig
{
    size_t trainingCount = 5000;
    size_t testCount = 10000;
    size_t trainingSteps = 40000;
    size_t evaluationInterval = 150;
    size_t gradientCheckSampleCount = 16;
    size_t imageWidth = 28;
    size_t imageHeight = 28;
};

#endif
