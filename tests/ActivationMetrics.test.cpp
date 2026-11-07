#include <gtest/gtest.h>

#include "Metrics/Metrics.hpp"

#include <limits>
#include <stdexcept>

TEST(ActivationMetricsTest, CountsSaturationAndSummarizesSigmoidDerivativeDistribution)
{
    std::array<float, 8> firstLayerValues{0.0f, 0.005f, 0.5f, 0.99f, 1.0f, 0.05f, 0.1f, 0.2f};
    std::array<float, 2> secondLayerValues{0.5f, 0.5f};
    std::array<float, 2> thirdLayerValues{0.0f, 1.0f};
    const std::array<float *, 3> activations{
        firstLayerValues.data(), secondLayerValues.data(), thirdLayerValues.data()};
    const std::array<size_t, 3> activationCounts{8, 2, 2};

    const Metrics::ActivationDiagnosticsCalculator calculator;
    const Metrics::ActivationDiagnostics diagnostics =
        calculator.calculate(activations, activationCounts);

    const Metrics::ActivationLayerMetrics &firstLayerMetrics = diagnostics.layers[0];
    EXPECT_EQ(firstLayerMetrics.activationCount, 8);
    EXPECT_EQ(firstLayerMetrics.saturatedCount, 3);
    EXPECT_DOUBLE_EQ(firstLayerMetrics.saturatedFraction, 0.375);
    EXPECT_NEAR(firstLayerMetrics.meanSigmoidDerivative, 0.070296875, 1e-8);
    EXPECT_EQ(firstLayerMetrics.derivativeBinCounts[0], 4);
    EXPECT_EQ(firstLayerMetrics.derivativeBinCounts[1], 1);
    EXPECT_EQ(firstLayerMetrics.derivativeBinCounts[2], 1);
    EXPECT_EQ(firstLayerMetrics.derivativeBinCounts[3], 1);
    EXPECT_EQ(firstLayerMetrics.derivativeBinCounts[4], 1);
    EXPECT_DOUBLE_EQ(firstLayerMetrics.derivativeBinFractions[0], 0.5);
    EXPECT_DOUBLE_EQ(firstLayerMetrics.derivativeBinFractions[4], 0.125);
    EXPECT_NEAR(firstLayerMetrics.derivativeBinFractions[0] +
                    firstLayerMetrics.derivativeBinFractions[1] +
                    firstLayerMetrics.derivativeBinFractions[2] +
                    firstLayerMetrics.derivativeBinFractions[3] +
                    firstLayerMetrics.derivativeBinFractions[4],
                1.0,
                1e-12);

    EXPECT_EQ(diagnostics.layers[1].saturatedCount, 0);
    EXPECT_DOUBLE_EQ(diagnostics.layers[1].meanSigmoidDerivative, 0.25);
    EXPECT_EQ(diagnostics.layers[2].saturatedCount, 2);
}

TEST(ActivationMetricsTest, RejectsInvalidActivationValues)
{
    float invalidActivation = std::numeric_limits<float>::quiet_NaN();
    float validActivation = 0.5f;
    const std::array<float *, 3> activations{
        &invalidActivation, &validActivation, &validActivation};
    const std::array<size_t, 3> activationCounts{1, 1, 1};

    const Metrics::ActivationDiagnosticsCalculator calculator;
    EXPECT_THROW(calculator.calculate(activations, activationCounts), std::invalid_argument);
}
