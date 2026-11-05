#include <gtest/gtest.h>

#include <VNN.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace
{
constexpr size_t sampleCount = 4;

EvaluationMetrics evaluateKnownLabels(NNG &network, const std::vector<float> &labels)
{
    std::vector<float> images(SIZE_INPUT_LAYER * sampleCount, 0.0f);
    return network.evaluate(images.data(), labels.data());
}
}

TEST(NNGEvaluationTest, ReportsConsistentLossAccuracyAndClassDiagnostics)
{
    NNG network(sampleCount);
    const std::vector<float> labels{0.0f, 1.0f, 1.0f, 9.0f};

    const EvaluationMetrics metrics = evaluateKnownLabels(network, labels);

    EXPECT_TRUE(std::isfinite(metrics.loss));
    EXPECT_GE(metrics.loss, 0.0f);
    EXPECT_GE(metrics.meanConfidence, 0.1f);
    EXPECT_LE(metrics.meanConfidence, 1.0f);

    EXPECT_EQ(std::accumulate(metrics.actualCounts.begin(), metrics.actualCounts.end(), size_t{0}),
              sampleCount);
    EXPECT_EQ(std::accumulate(metrics.predictedCounts.begin(), metrics.predictedCounts.end(), size_t{0}),
              sampleCount);
    EXPECT_EQ(std::accumulate(metrics.confusionCounts.begin(), metrics.confusionCounts.end(), size_t{0}),
              sampleCount);

    size_t correctCount = 0;
    for (size_t actual = 0; actual < 10; ++actual)
    {
        correctCount += metrics.confusionCounts[actual * 10 + actual];
        size_t rowCount = 0;
        for (size_t predicted = 0; predicted < 10; ++predicted)
        {
            const size_t index = actual * 10 + predicted;
            rowCount += metrics.confusionCounts[index];
            if (metrics.actualCounts[actual] == 0)
            {
                EXPECT_DOUBLE_EQ(metrics.normalizedConfusion[index], 0.0);
            }
            else
            {
                EXPECT_DOUBLE_EQ(
                    metrics.normalizedConfusion[index],
                    static_cast<double>(metrics.confusionCounts[index]) /
                        static_cast<double>(metrics.actualCounts[actual]));
            }
        }
        EXPECT_EQ(rowCount, metrics.actualCounts[actual]);
        if (metrics.actualCounts[actual] > 0)
        {
            EXPECT_FLOAT_EQ(
                metrics.classRecall[actual],
                static_cast<float>(metrics.confusionCounts[actual * 10 + actual]) /
                    static_cast<float>(metrics.actualCounts[actual]));
        }
        else
        {
            EXPECT_FLOAT_EQ(metrics.classRecall[actual], 0.0f);
        }
    }

    EXPECT_FLOAT_EQ(metrics.accuracy, static_cast<float>(correctCount) / sampleCount);
    EXPECT_EQ(metrics.actualCounts[0], 1);
    EXPECT_EQ(metrics.actualCounts[1], 2);
    EXPECT_EQ(metrics.actualCounts[9], 1);
}

TEST(NNGEvaluationTest, RejectsInvalidTestLabels)
{
    NNG network(1);
    const std::vector<float> image(SIZE_INPUT_LAYER, 0.0f);
    const std::vector<float> invalidLabels{10.0f};

    EXPECT_THROW(network.evaluate(image.data(), invalidLabels.data()), std::invalid_argument);
}

TEST(NNGEvaluationTest, RejectsMissingEvaluationInputs)
{
    NNG network(1);
    const std::vector<float> label{0.0f};

    EXPECT_THROW(network.evaluate(nullptr, label.data()), std::invalid_argument);
    EXPECT_THROW(network.evaluate(label.data(), nullptr), std::invalid_argument);
}

TEST(NNGGradientCheckTest, MatchesAnalyticalGradientsAgainstFiniteDifferences)
{
    constexpr size_t count = 4;
    NNG network(1);
    std::vector<float> images(SIZE_INPUT_LAYER * count);
    for (size_t feature = 0; feature < SIZE_INPUT_LAYER; ++feature)
    {
        for (size_t sample = 0; sample < count; ++sample)
        {
            images[feature * count + sample] =
                static_cast<float>((feature * (sample + 1)) % 255) / 255.0f;
        }
    }
    const std::vector<float> labels{0.0f, 1.0f, 3.0f, 9.0f};

    EXPECT_TRUE(network.runGradientCheck(images.data(), labels.data(), count));
}
