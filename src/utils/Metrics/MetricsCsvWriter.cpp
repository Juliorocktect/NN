#include "MetricsCsvWriter.hpp"

#include "Metrics/Metrics.hpp"

#include <iomanip>
#include <limits>
#include <stdexcept>

MetricsCsvWriter::MetricsCsvWriter(const std::string &filePath)
    : file(filePath, std::ios::out | std::ios::trunc)
{
    if (!file.is_open())
        throw std::runtime_error("Could not open metrics CSV file: " + filePath);

    writeHeader();
}

void MetricsCsvWriter::writeHeader()
{
    file << "step,training_loss,test_loss,accuracy,mean_confidence";
    for (size_t category = 0; category < 10; ++category)
        file << ",actual_count_" << category;
    for (size_t category = 0; category < 10; ++category)
        file << ",predicted_count_" << category;
    for (size_t category = 0; category < 10; ++category)
        file << ",class_recall_" << category;
    for (size_t actual = 0; actual < 10; ++actual)
        for (size_t predicted = 0; predicted < 10; ++predicted)
            file << ",confusion_count_" << actual << '_' << predicted;
    for (size_t actual = 0; actual < 10; ++actual)
        for (size_t predicted = 0; predicted < 10; ++predicted)
            file << ",normalized_confusion_" << actual << '_' << predicted;
    for (size_t layer = 0; layer < 3; ++layer)
    {
        file << ",hidden_layer_" << layer + 1 << "_activation_count"
             << ",hidden_layer_" << layer + 1 << "_saturated_count"
             << ",hidden_layer_" << layer + 1 << "_saturated_fraction"
             << ",hidden_layer_" << layer + 1 << "_mean_sigmoid_derivative";
        for (size_t bin = 0; bin < Metrics::ACTIVATION_DERIVATIVE_BIN_COUNT; ++bin)
            file << ",hidden_layer_" << layer + 1 << "_derivative_bin_" << bin << "_fraction";
    }
    file << '\n';
    file.flush();

    if (!file)
        throw std::runtime_error("Could not write metrics CSV header");
}

void MetricsCsvWriter::writeMetrics(
    size_t step,
    float trainingLoss,
    const EvaluationMetrics &metrics)
{
    writeMetrics(step, trainingLoss, metrics, Metrics::ActivationDiagnostics{});
}

void MetricsCsvWriter::writeMetrics(
    size_t step,
    float trainingLoss,
    const EvaluationMetrics &metrics,
    const Metrics::ActivationDiagnostics &activationDiagnostics)
{
    file << std::setprecision(std::numeric_limits<double>::max_digits10)
         << step << ','
         << trainingLoss << ','
         << metrics.loss << ','
         << metrics.accuracy << ','
         << metrics.meanConfidence;

    for (size_t count : metrics.actualCounts)
        file << ',' << count;
    for (size_t count : metrics.predictedCounts)
        file << ',' << count;
    for (float recall : metrics.classRecall)
        file << ',' << recall;
    for (size_t count : metrics.confusionCounts)
        file << ',' << count;
    for (double value : metrics.normalizedConfusion)
        file << ',' << value;
    for (const Metrics::ActivationLayerMetrics &layer : activationDiagnostics.layers)
    {
        file << ',' << layer.activationCount
             << ',' << layer.saturatedCount
             << ',' << layer.saturatedFraction
             << ',' << layer.meanSigmoidDerivative;
        for (double fraction : layer.derivativeBinFractions)
            file << ',' << fraction;
    }

    file << '\n';
    file.flush();

    if (!file)
        throw std::runtime_error("Could not write metrics CSV row");
}
