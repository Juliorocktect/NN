#ifndef METRICS_CSV_WRITER_HPP
#define METRICS_CSV_WRITER_HPP

#include <cstddef>
#include <fstream>
#include <string>

#include "Metrics/EvaluationMetrics.hpp"
namespace Metrics
{
struct ActivationDiagnostics;
}

class MetricsCsvWriter
{
public:
    explicit MetricsCsvWriter(const std::string &filePath);

    MetricsCsvWriter(const MetricsCsvWriter &) = delete;
    MetricsCsvWriter &operator=(const MetricsCsvWriter &) = delete;

    void writeMetrics(size_t step, float trainingLoss, const EvaluationMetrics &metrics);
    void writeMetrics(
        size_t step,
        float trainingLoss,
        const EvaluationMetrics &metrics,
        const Metrics::ActivationDiagnostics &activationDiagnostics);

private:
    void writeHeader();

    std::ofstream file;
};

#endif
