#ifndef METRICS_EVALUATION_METRICS_HPP
#define METRICS_EVALUATION_METRICS_HPP

#include <array>
#include <cstddef>

/**
 * @brief Aggregated classification metrics for one evaluation dataset.
 */
struct EvaluationMetrics
{
    /** @brief Mean cross-entropy loss over all evaluated samples. */
    float loss = 0.0f;
    /** @brief Fraction of samples classified correctly, in the range [0, 1]. */
    float accuracy = 0.0f;
    /** @brief Mean probability assigned to each sample's predicted class. */
    float meanConfidence = 0.0f;
    /** @brief Number of samples for each actual class, indexed by class label. */
    std::array<size_t, 10> actualCounts{};
    /** @brief Number of samples predicted for each class, indexed by predicted label. */
    std::array<size_t, 10> predictedCounts{};
    /** @brief Confusion matrix counts, indexed as actual class * 10 + predicted class. */
    std::array<size_t, 100> confusionCounts{};
    /** @brief Row-normalized confusion matrix, indexed as actual class * 10 + predicted class. */
    std::array<double, 100> normalizedConfusion{};
    /** @brief Recall for each actual class; zero when that class has no samples. */
    std::array<float, 10> classRecall{};
};

#endif
