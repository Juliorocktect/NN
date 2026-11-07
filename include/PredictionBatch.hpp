#ifndef PREDICTION_BATCH_HPP
#define PREDICTION_BATCH_HPP

#include <cstddef>
#include <vector>

struct PredictionBatch
{
    std::vector<size_t> predictedClasses;
    std::vector<float> confidenceValues;
};

#endif
