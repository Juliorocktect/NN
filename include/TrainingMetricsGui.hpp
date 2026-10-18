#ifndef TRAINING_METRICS_GUI_HPP
#define TRAINING_METRICS_GUI_HPP

#include <SDL3/SDL.h>
#include <array>
#include <cstddef>
#include <vector>

class TrainingMetricsGui
{
public:
    TrainingMetricsGui();
    ~TrainingMetricsGui();

    TrainingMetricsGui(const TrainingMetricsGui &) = delete;
    TrainingMetricsGui &operator=(const TrainingMetricsGui &) = delete;

    bool processEvents();
    void addMetrics(size_t step, float trainingLoss, float testLoss, float testAccuracy);
    void setDiagnostics(
        const std::array<size_t, 10> &actualCounts,
        const std::array<size_t, 10> &predictedCounts,
        const std::array<size_t, 100> &confusionCounts,
        const std::array<double, 100> &normalizedConfusion,
        const std::array<float, 10> &classRecall,
        float meanConfidence);
    void render();

private:
    void shutdown() noexcept;

    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;
    bool sdlInitialized = false;
    bool imguiInitialized = false;
    bool imguiSdlBackendInitialized = false;
    bool imguiRendererBackendInitialized = false;
    bool running = true;
    std::vector<double> steps;
    std::vector<double> trainingLosses;
    std::vector<double> testLosses;
    std::vector<double> testAccuracies;
    std::array<double, 10> classPositions{};
    std::array<double, 10> predictedCounts{};
    std::array<double, 100> normalizedConfusion{};
    std::array<double, 10> classRecall{};
    std::array<size_t, 10> actualCounts{};
    std::array<size_t, 100> confusionCounts{};
    double meanConfidence = 0.0;
    bool hasDiagnostics = false;
};

#endif
