#include <gtest/gtest.h>

#include "TrainingController.hpp"
#include "VNN.h"

#include <stdexcept>

TEST(TrainingControllerTest, RejectsInvalidDatasetBeforeStartingTraining)
{
    TrainingController trainingController;
    TrainingDataset dataset;
    const TrainingConfig config;
    const ConvolutionConfig convolutionConfig;

    EXPECT_THROW(
        trainingController.run(dataset, config, convolutionConfig),
        std::invalid_argument);
}
