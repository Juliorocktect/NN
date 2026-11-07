#include <gtest/gtest.h>

#include "TrainingController.hpp"
#include "VNN.h"

#include <stdexcept>

TEST(TrainingControllerTest, RejectsInvalidDatasetBeforeStartingTraining)
{
    TrainingController trainingController;
    TrainingDataset dataset;
    const TrainingConfig config;

    EXPECT_THROW(
        trainingController.run(dataset, config),
        std::invalid_argument);
}
