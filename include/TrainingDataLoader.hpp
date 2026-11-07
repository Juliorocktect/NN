#ifndef TRAINING_DATA_LOADER_HPP
#define TRAINING_DATA_LOADER_HPP

#include "TrainingConfig.hpp"
#include "TrainingDataset.hpp"

class TrainingDataLoader
{
public:
    TrainingDataset load(const TrainingConfig &config) const;
};

#endif
