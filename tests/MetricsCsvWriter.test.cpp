#include <gtest/gtest.h>

#include "MetricsCsvWriter.hpp"
#include "VNN.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
std::vector<std::string> splitCsvRow(const std::string &row)
{
    std::vector<std::string> fields;
    std::stringstream stream(row);
    std::string field;
    while (std::getline(stream, field, ','))
        fields.push_back(field);
    return fields;
}

std::filesystem::path createTemporaryCsvPath()
{
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("nn_metrics_" + std::to_string(timestamp) + ".csv");
}
}

TEST(MetricsCsvWriterTest, WritesTrainingAndEvaluationMetricsToCsv)
{
    const std::filesystem::path path = createTemporaryCsvPath();
    EvaluationMetrics metrics;
    metrics.loss = 0.25f;
    metrics.accuracy = 0.75f;
    metrics.meanConfidence = 0.9f;
    metrics.actualCounts[2] = 2;
    metrics.predictedCounts[2] = 1;
    metrics.predictedCounts[3] = 1;
    metrics.classRecall[2] = 0.5f;
    metrics.confusionCounts[2 * 10 + 2] = 1;
    metrics.confusionCounts[2 * 10 + 3] = 1;
    metrics.normalizedConfusion[2 * 10 + 2] = 0.5;
    metrics.normalizedConfusion[2 * 10 + 3] = 0.5;

    {
        MetricsCsvWriter writer(path.string());
        writer.writeMetrics(100, 0.3f, metrics);
    }

    std::ifstream csv(path);
    ASSERT_TRUE(csv.is_open());
    std::string header;
    std::string row;
    ASSERT_TRUE(std::getline(csv, header));
    ASSERT_TRUE(std::getline(csv, row));

    const std::vector<std::string> headers = splitCsvRow(header);
    const std::vector<std::string> values = splitCsvRow(row);
    ASSERT_EQ(headers.size(), 235);
    ASSERT_EQ(values.size(), headers.size());
    EXPECT_EQ(values[0], "100");
    EXPECT_FLOAT_EQ(std::stof(values[1]), 0.3f);
    EXPECT_FLOAT_EQ(std::stof(values[2]), metrics.loss);
    EXPECT_FLOAT_EQ(std::stof(values[3]), metrics.accuracy);
    EXPECT_FLOAT_EQ(std::stof(values[4]), metrics.meanConfidence);
    EXPECT_EQ(headers[5 + 2], "actual_count_2");
    EXPECT_EQ(values[5 + 2], "2");
    EXPECT_EQ(headers[25 + 2], "class_recall_2");
    EXPECT_FLOAT_EQ(std::stof(values[25 + 2]), 0.5f);

    csv.close();
    std::filesystem::remove(path);
}
