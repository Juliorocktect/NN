#ifndef PREDICTION_BATCH_HPP
#define PREDICTION_BATCH_HPP

#include <cstddef>
#include <vector>

/**
 * @brief Per-sample predictions returned by a network.
 */
struct PredictionBatch
{
    /** @brief Predicted class label for each sample. */
    std::vector<size_t> predictedClasses;
    /** @brief Probability assigned to each sample's predicted class. */
    std::vector<float> confidenceValues;
};

#endif
