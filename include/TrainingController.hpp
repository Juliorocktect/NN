#ifndef TRAINING_CONTROLLER_HPP
#define TRAINING_CONTROLLER_HPP

#include "TrainingConfig.hpp"
#include "TrainingDataset.hpp"

class TrainingController
{
public:
    void run(TrainingDataset &dataset, const TrainingConfig &config) const;

private:
    struct TrainingState;

    static bool runTrainingStep(void *userData);
};

#endif
