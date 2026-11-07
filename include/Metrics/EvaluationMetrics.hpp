#ifndef METRICS_EVALUATION_METRICS_HPP
#define METRICS_EVALUATION_METRICS_HPP

#include <array>
#include <cstddef>

struct EvaluationMetrics
{
    float loss = 0.0f;
    float accuracy = 0.0f;
    float meanConfidence = 0.0f;
    std::array<size_t, 10> actualCounts{};
    std::array<size_t, 10> predictedCounts{};
    std::array<size_t, 100> confusionCounts{};
    std::array<double, 100> normalizedConfusion{};
    std::array<float, 10> classRecall{};
};

#endif
