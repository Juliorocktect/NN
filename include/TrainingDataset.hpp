#ifndef TRAINING_DATASET_HPP
#define TRAINING_DATASET_HPP

#include <cstddef>
#include <memory>
#include <vector>

/**
 * @brief Owns the loaded training and test samples and their labels.
 */
struct TrainingDataset
{
    /** @brief Labels corresponding to trainingImages. */
    std::vector<float> trainingLabels;
    /** @brief Labels corresponding to testImages. */
    std::vector<float> testLabels;
    /** @brief Training images in feature-major, sample-minor layout. */
    std::unique_ptr<float[]> trainingImages;
    /** @brief Test images in feature-major, sample-minor layout. */
    std::unique_ptr<float[]> testImages;
    /** @brief Number of training images actually loaded. */
    size_t trainingCount = 0;
    /** @brief Number of test images actually loaded. */
    size_t testCount = 0;
};

#endif
