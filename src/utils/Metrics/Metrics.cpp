#include "Metrics/Metrics.hpp"

#include <iostream>

namespace Metrics
{
bool trainingStepCallback(void *userData)
{
    TrainingState *state = static_cast<TrainingState *>(userData);
    ++state->step;
    const float trainingLoss = state->network->trainStep();
    if (state->step % state->evaluationInterval == 0 ||
        state->step == 1 ||
        state->step == state->trainingSteps)
    {
        state->evaluator->copyParametersFrom(*state->network);
        const EvaluationMetrics metrics =
            state->evaluator->evaluate(state->testImages, state->testLabels);
        state->metricsCsv->writeMetrics(state->step, trainingLoss, metrics);
        state->gui->queueMetrics(
            state->step,
            trainingLoss,
            metrics.loss,
            metrics.accuracy,
            metrics.actualCounts,
            metrics.predictedCounts,
            metrics.confusionCounts,
            metrics.normalizedConfusion,
            metrics.classRecall,
            metrics.meanConfidence);
        std::cout << "Schritt " << state->step
                  << " | Trainings-Loss: " << trainingLoss
                  << " | Test-Loss: " << metrics.loss
                  << " | Testgenauigkeit: " << metrics.accuracy
                  << " | Vorhersagen je Klasse:";
        for (size_t category = 0; category < metrics.predictedCounts.size(); ++category)
            std::cout << ' ' << category << '=' << metrics.predictedCounts[category];
        std::cout << '\n';
    }

    if (state->step == state->trainingSteps)
    {
        constexpr size_t GRADIENT_CHECK_SAMPLE_COUNT = 16;
        const bool passed = state->network->runGradientCheck(
            state->gradientCheckImages.data(),
            state->trainingLabels,
            GRADIENT_CHECK_SAMPLE_COUNT);
        if (!passed)
            std::cerr << "Warnung: Gradiententest am Trainingsende ist fehlgeschlagen.\n";
        std::cout.flush();
        std::cerr.flush();
        return false;
    }

    return true;
}
}
