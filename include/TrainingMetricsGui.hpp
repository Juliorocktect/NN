#ifndef TRAINING_METRICS_GUI_HPP
#define TRAINING_METRICS_GUI_HPP

#include <gtk/gtk.h>

#include <array>
#include <cstddef>
#include <exception>
#include <functional>
#include <vector>

class TrainingMetricsGui
{
public:
    TrainingMetricsGui();
    ~TrainingMetricsGui();

    TrainingMetricsGui(const TrainingMetricsGui &) = delete;
    TrainingMetricsGui &operator=(const TrainingMetricsGui &) = delete;

    void setTrainingStepCallback(std::function<bool()> callback);
    void run();
    void rethrowTrainingError() const;
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
    static gboolean onCloseRequest(GtkWindow *window, gpointer userData);
    static void onStartClicked(GtkButton *button, gpointer userData);
    static void onPauseClicked(GtkButton *button, gpointer userData);
    static void onStopClicked(GtkButton *button, gpointer userData);
    static gboolean onTrainingStep(gpointer userData);
    static void drawCallback(
        GtkDrawingArea *area,
        cairo_t *context,
        int width,
        int height,
        gpointer userData);
    void draw(cairo_t *context, int width, int height) const;
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
    GMainLoop *mainLoop = nullptr;
    guint trainingSourceId = 0;
    bool trainingStarted = false;
    bool trainingActive = false;
    bool trainingStopped = false;
    bool trainingCompleted = false;
    std::function<bool()> trainingStepCallback;
    std::exception_ptr trainingError;
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
