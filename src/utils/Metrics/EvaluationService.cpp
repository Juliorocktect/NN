#include "EvaluationService.hpp"

#include "VNN.h"

#include <cmath>
#include <stdexcept>

EvaluationMetrics EvaluationService::evaluate(
    NNG &network,
    const float *testImages,
    const float *testLabels,
    size_t sampleCount) const
{
    if (testImages == nullptr || testLabels == nullptr || sampleCount == 0)
    {
        throw std::invalid_argument("Evaluation requires non-empty test data");
    }

    for (size_t sample = 0; sample < sampleCount; ++sample)
    {
        const float labelValue = testLabels[sample];
        if (!std::isfinite(labelValue) || labelValue < 0.0f || labelValue > 9.0f)
        {
            throw std::invalid_argument("Test labels must be finite integers from 0 to 9");
        }

        const int label = static_cast<int>(labelValue);
        if (label < 0 || label >= 10 || labelValue != label)
        {
            throw std::invalid_argument("Test labels must be integers from 0 to 9");
        }
    }

    const PredictionBatch predictions = network.predict(testImages);
    if (predictions.predictedClasses.size() != sampleCount ||
        predictions.confidenceValues.size() != sampleCount)
    {
        throw std::invalid_argument("Network prediction count does not match test data");
    }

    EvaluationMetrics metrics;
    metrics.loss = network.calculateEvaluationLoss(testLabels);

    float totalConfidence = 0.0f;
    size_t correctCount = 0;
    for (size_t sample = 0; sample < sampleCount; ++sample)
    {
        const size_t actualLabel = static_cast<size_t>(testLabels[sample]);
        const size_t predictedLabel = predictions.predictedClasses[sample];
        ++metrics.actualCounts[actualLabel];
        ++metrics.predictedCounts[predictedLabel];
        ++metrics.confusionCounts[actualLabel * metrics.predictedCounts.size() + predictedLabel];
        if (actualLabel == predictedLabel)
        {
            ++correctCount;
        }
        totalConfidence += predictions.confidenceValues[sample];
    }

    metrics.accuracy = static_cast<float>(correctCount) / static_cast<float>(sampleCount);
    metrics.meanConfidence = totalConfidence / static_cast<float>(sampleCount);
    calculateClassMetrics(metrics);
    return metrics;
}

void EvaluationService::calculateClassMetrics(EvaluationMetrics &metrics) const
{
    for (size_t actual = 0; actual < metrics.actualCounts.size(); ++actual)
    {
        const size_t actualCount = metrics.actualCounts[actual];
        if (actualCount == 0)
        {
            continue;
        }

        const size_t classCorrectCount =
            metrics.confusionCounts[actual * metrics.predictedCounts.size() + actual];
        metrics.classRecall[actual] =
            static_cast<float>(classCorrectCount) / static_cast<float>(actualCount);

        for (size_t predicted = 0; predicted < metrics.predictedCounts.size(); ++predicted)
        {
            const size_t index = actual * metrics.predictedCounts.size() + predicted;
            metrics.normalizedConfusion[index] =
                static_cast<double>(metrics.confusionCounts[index]) /
                static_cast<double>(actualCount);
        }
    }
}
