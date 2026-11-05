#ifndef METRICS_CSV_WRITER_HPP
#define METRICS_CSV_WRITER_HPP

#include <cstddef>
#include <fstream>
#include <string>

struct EvaluationMetrics;

class MetricsCsvWriter
{
public:
    explicit MetricsCsvWriter(const std::string &filePath);

    MetricsCsvWriter(const MetricsCsvWriter &) = delete;
    MetricsCsvWriter &operator=(const MetricsCsvWriter &) = delete;

    void writeMetrics(size_t step, float trainingLoss, const EvaluationMetrics &metrics);

private:
    void writeHeader();

    std::ofstream file;
};

#endif
