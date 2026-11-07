#ifndef EVALUATION_SERVICE_HPP
#define EVALUATION_SERVICE_HPP

#include "Metrics/EvaluationMetrics.hpp"

#include <cstddef>

class NNG;

/**
 * @brief Evaluates network predictions and aggregates classification metrics.
 */
class EvaluationService
{
public:
    /**
     * @brief Predicts the test samples and computes loss, accuracy, confidence, and class metrics.
     *
     * @param network Network to evaluate; its configured batch size must equal sampleCount.
     * @param testImages Input images in the network's column-major batch layout.
     * @param testLabels Expected integer class labels in the range [0, 9].
     * @param sampleCount Number of test samples.
     * @return Aggregated evaluation metrics for the test samples.
     * @throws std::invalid_argument If the inputs, labels, or prediction count are invalid.
     */
    EvaluationMetrics evaluate(
        NNG &network,
        const float *testImages,
        const float *testLabels,
        size_t sampleCount) const;

private:
    /**
     * @brief Calculates per-class recall and row-normalized confusion values.
     *
     * @param metrics Metrics containing the actual counts and confusion matrix to normalize.
     */
    void calculateClassMetrics(EvaluationMetrics &metrics) const;
};

#endif
