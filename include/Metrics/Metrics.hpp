#ifndef METRICS_METRICS_HPP
#define METRICS_METRICS_HPP

#include "MetricsCsvWriter.hpp"
#include "TrainingMetricsGui.hpp"
#include "VNN.h"

#include <cstddef>
#include <vector>

namespace Metrics
{
struct TrainingState
{
    NNG *network;
    NNG *evaluator;
    TrainingMetricsGui *gui;
    MetricsCsvWriter *metricsCsv;
    const float *testImages;
    const float *testLabels;
    const float *trainingLabels;
    size_t trainingSteps;
    size_t evaluationInterval;
    size_t step = 0;
    std::vector<float> gradientCheckImages;
};

bool trainingStepCallback(void *userData);
}

#endif
