#ifndef TRAINING_MANAGER_HPP
#define TRAINING_MANAGER_HPP

#include <cstddef>
#include <vector>

class NNG;

class TrainingManager
{
public:
    void run(
        const float *trainingImages,
        std::vector<float> &trainingLabels,
        const float *testImages,
        const std::vector<float> &testLabels,
        size_t trainingCount,
        size_t testCount) const;
    float test(
        NNG &network,
        NNG &evaluator,
        const float *testImages,
        const float *testLabels,
        size_t testCount) const;
};

#endif
