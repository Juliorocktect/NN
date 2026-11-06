#include <gtest/gtest.h>

#include "MetricsCsvWriter.hpp"
#include "Metrics/Metrics.hpp"
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
    Metrics::ActivationDiagnostics activationDiagnostics;
    activationDiagnostics.layers[0].activationCount = 100;
    activationDiagnostics.layers[0].saturatedCount = 20;
    activationDiagnostics.layers[0].saturatedFraction = 0.2;
    activationDiagnostics.layers[0].meanSigmoidDerivative = 0.12;
    activationDiagnostics.layers[0].derivativeBinFractions = {0.1, 0.2, 0.3, 0.25, 0.15};

    {
        MetricsCsvWriter writer(path.string());
        writer.writeMetrics(100, 0.3f, metrics, activationDiagnostics);
    }

    std::ifstream csv(path);
    ASSERT_TRUE(csv.is_open());
    std::string header;
    std::string row;
    ASSERT_TRUE(std::getline(csv, header));
    ASSERT_TRUE(std::getline(csv, row));

    const std::vector<std::string> headers = splitCsvRow(header);
    const std::vector<std::string> values = splitCsvRow(row);
    ASSERT_EQ(headers.size(), 262);
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
    EXPECT_EQ(headers[235], "hidden_layer_1_activation_count");
    EXPECT_EQ(values[235], "100");
    EXPECT_EQ(headers[236], "hidden_layer_1_saturated_count");
    EXPECT_EQ(values[236], "20");
    EXPECT_EQ(headers[237], "hidden_layer_1_saturated_fraction");
    EXPECT_DOUBLE_EQ(std::stod(values[237]), 0.2);
    EXPECT_EQ(headers[238], "hidden_layer_1_mean_sigmoid_derivative");
    EXPECT_DOUBLE_EQ(std::stod(values[238]), 0.12);
    EXPECT_EQ(headers[239], "hidden_layer_1_derivative_bin_0_fraction");
    EXPECT_DOUBLE_EQ(std::stod(values[239]), 0.1);
    EXPECT_EQ(headers[261], "hidden_layer_3_derivative_bin_4_fraction");
    EXPECT_DOUBLE_EQ(std::stod(values[261]), 0.0);

    csv.close();
    std::filesystem::remove(path);
}
