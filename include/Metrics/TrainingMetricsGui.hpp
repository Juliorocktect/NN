#ifndef TRAINING_METRICS_GUI_HPP
#define TRAINING_METRICS_GUI_HPP

#include "Metrics/Metrics.hpp"

#include <gtk/gtk.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

class TrainingMetricsGui
{
public:
    using TrainingStepCallback = bool (*)(void *);

    TrainingMetricsGui();
    ~TrainingMetricsGui();

    TrainingMetricsGui(const TrainingMetricsGui &) = delete;
    TrainingMetricsGui &operator=(const TrainingMetricsGui &) = delete;

    void setTrainingStepCallback(TrainingStepCallback callback, void *userData);
    void setImageData(
        const float *images,
        const float *labels,
        size_t imageCount,
        size_t imageWidth,
        size_t imageHeight);
    void run();
    void rethrowTrainingError() const;
    void queueMetrics(const Metrics::TrainingMetricsSnapshot &metrics);
    void queueFinalTestAccuracy(float accuracy);
    void render();

private:
    static gboolean onCloseRequest(GtkWindow *window, gpointer userData);
    static void onStartClicked(GtkButton *button, gpointer userData);
    static void onPauseClicked(GtkButton *button, gpointer userData);
    static void onStopClicked(GtkButton *button, gpointer userData);
    static void onImageLookupClicked(GtkButton *button, gpointer userData);
    static gboolean onUiUpdate(gpointer userData);
    static void drawCallback(
        GtkDrawingArea *area,
        cairo_t *context,
        int width,
        int height,
        gpointer userData);
    static void imagePreviewDrawCallback(
        GtkDrawingArea *area,
        cairo_t *context,
        int width,
        int height,
        gpointer userData);
    void draw(cairo_t *context, int width, int height) const;
    void drawImagePreview(cairo_t *context, int width, int height) const;
    void showSelectedImage();
    void runTrainingWorker();
    void applyQueuedMetrics();
    void startTraining();
    void pauseTraining();
    void stopTraining();
    void updateControls();

    GtkWindow *window = nullptr;
    GtkWidget *drawingArea = nullptr;
    GtkWidget *startButton = nullptr;
    GtkWidget *pauseButton = nullptr;
    GtkWidget *stopButton = nullptr;
    GtkWidget *statusLabel = nullptr;
    GtkWidget *imageIndexEntry = nullptr;
    GtkWidget *imageLookupStatus = nullptr;
    GtkWidget *imagePreviewArea = nullptr;
    GMainLoop *mainLoop = nullptr;
    guint uiUpdateSourceId = 0;
    bool trainingStarted = false;
    bool trainingActive = false;
    bool trainingStopped = false;
    bool trainingCompleted = false;
    const float *imageData = nullptr;
    const float *imageLabels = nullptr;
    size_t imageCount = 0;
    size_t imageWidth = 0;
    size_t imageHeight = 0;
    size_t selectedImageIndex = 0;
    bool hasSelectedImage = false;
    TrainingStepCallback trainingStepCallback = nullptr;
    void *trainingStepUserData = nullptr;
    std::exception_ptr trainingError;
    std::thread trainingThread;
    std::mutex trainingMutex;
    std::condition_variable trainingCondition;
    bool workerActive = false;
    bool workerStopped = false;
    std::atomic<bool> workerFinished{false};
    std::vector<Metrics::TrainingMetricsSnapshot> pendingMetrics;
    std::optional<float> pendingFinalTestAccuracy;
    std::optional<float> finalTestAccuracy;
    std::vector<Metrics::ActivationHistoryEntry> activationHistory;
    std::vector<double> steps;
    std::vector<double> trainingLosses;
    std::vector<double> testLosses;
    std::vector<double> testAccuracies;
    std::array<double, 10> predictedCounts{};
    std::array<double, 100> normalizedConfusion{};
    std::array<double, 10> classRecall{};
    std::array<size_t, 10> actualCounts{};
    std::array<size_t, 100> confusionCounts{};
    double meanConfidence = 0.0;
    bool hasDiagnostics = false;
};

#endif
