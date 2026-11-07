#ifndef EVALUATION_SERVICE_HPP
#define EVALUATION_SERVICE_HPP

#include "Metrics/EvaluationMetrics.hpp"

#include <cstddef>

class NNG;

class EvaluationService
{
public:
    EvaluationMetrics evaluate(
        NNG &network,
        const float *testImages,
        const float *testLabels,
        size_t sampleCount) const;

private:
    void calculateClassMetrics(EvaluationMetrics &metrics) const;
};

#endif
