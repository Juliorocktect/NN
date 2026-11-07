#include <gtest/gtest.h>

#include "TrainingManager.hpp"
#include "VNN.h"

#include <stdexcept>
#include <vector>

TEST(TrainingManagerTest, RejectsInvalidTrainingDataBeforeStartingTraining)
{
    TrainingManager trainingManager;
    std::vector<float> labels;

    EXPECT_THROW(
        trainingManager.run(nullptr, labels, nullptr, labels, 0, 0),
        std::invalid_argument);
}

TEST(TrainingManagerTest, RejectsInvalidTestData)
{
    TrainingManager trainingManager;
    NNG network(1);
    NNG evaluator(1);
    const float label = 0.0f;

    EXPECT_THROW(
        trainingManager.test(network, evaluator, nullptr, &label, 1),
        std::invalid_argument);
}
