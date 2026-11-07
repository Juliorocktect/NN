#ifndef TRAINING_DATA_LOADER_HPP
#define TRAINING_DATA_LOADER_HPP

#include "TrainingConfig.hpp"
#include "TrainingDataset.hpp"

/**
 * @brief Loads and validates image and label data for a training run.
 */
class TrainingDataLoader
{
public:
    /**
     * @brief Loads the configured number of training and test samples.
     *
     * @param config Requested training and test sample counts.
     * @return Dataset owning the loaded images and labels.
     * @throws std::invalid_argument If a requested sample count is zero.
     * @throws std::runtime_error If labels or images cannot be loaded as requested.
     */
    TrainingDataset load(const TrainingConfig &config) const;
};

#endif
