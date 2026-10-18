#ifndef TRAINING_METRICS_GUI_HPP
#define TRAINING_METRICS_GUI_HPP

#include <SDL3/SDL.h>
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
};

#endif
