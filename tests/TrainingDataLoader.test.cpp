#include <gtest/gtest.h>

#include "TrainingDataLoader.hpp"

#include <stdexcept>

TEST(TrainingDataLoaderTest, RejectsZeroTrainingCount)
{
    TrainingDataLoader dataLoader;
    TrainingConfig config;
    config.trainingCount = 0;

    EXPECT_THROW(dataLoader.load(config), std::invalid_argument);
}
