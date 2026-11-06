#include "TrainingMetricsGui.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace
{
constexpr double margin = 48.0;
constexpr double titleHeight = 28.0;

void setColor(cairo_t *context, double red, double green, double blue, double alpha = 1.0)
{
    cairo_set_source_rgba(context, red, green, blue, alpha);
}

void drawText(cairo_t *context, double x, double y, const char *text, double size = 13.0)
{
    cairo_select_font_face(context, "sans-serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(context, size);
    cairo_move_to(context, x, y);
    cairo_show_text(context, text);
}

void drawChartFrame(cairo_t *context, double x, double y, double width, double height, const char *title)
{
    setColor(context, 0.14, 0.16, 0.20);
    cairo_rectangle(context, x, y, width, height);
    cairo_fill(context);

    setColor(context, 0.88, 0.90, 0.94);
    drawText(context, x + 14.0, y + 20.0, title, 15.0);

    setColor(context, 0.32, 0.35, 0.40);
    cairo_set_line_width(context, 1.0);
    cairo_rectangle(context, x + margin, y + titleHeight, width - margin - 14.0, height - titleHeight - 32.0);
    cairo_stroke(context);
}

void drawLineSeries(
    cairo_t *context,
    double x,
    double y,
    double width,
    double height,
    const std::vector<double> &values,
    double minimum,
    double maximum,
    double red,
    double green,
    double blue)
{
    if (values.empty())
        return;

    const double plotX = x + margin;
    const double plotY = y + titleHeight;
    const double plotWidth = std::max(1.0, width - margin - 14.0);
    const double plotHeight = std::max(1.0, height - titleHeight - 32.0);
    const double range = std::max(1e-12, maximum - minimum);

    setColor(context, red, green, blue);
    cairo_set_line_width(context, 2.0);
    for (size_t index = 0; index < values.size(); ++index)
    {
        const double px = plotX + (values.size() == 1
                                       ? 0.0
                                       : plotWidth * static_cast<double>(index) / static_cast<double>(values.size() - 1));
        const double normalized = std::clamp((values[index] - minimum) / range, 0.0, 1.0);
        const double py = plotY + plotHeight * (1.0 - normalized);
        if (index == 0)
            cairo_move_to(context, px, py);
        else
            cairo_line_to(context, px, py);
    }
    cairo_stroke(context);
}

void drawBarChart(
    cairo_t *context,
    double x,
    double y,
    double width,
    double height,
    const char *title,
    const std::array<double, 10> &values,
    double maximum,
    bool percentage)
{
    drawChartFrame(context, x, y, width, height, title);
    const double plotX = x + margin;
    const double plotY = y + titleHeight;
    const double plotWidth = std::max(1.0, width - margin - 14.0);
    const double plotHeight = std::max(1.0, height - titleHeight - 32.0);
    const double barWidth = plotWidth / 10.0;
    const double valueRange = std::max(maximum, 1e-12);

    for (size_t category = 0; category < values.size(); ++category)
    {
        const double barHeight = plotHeight * std::clamp(values[category] / valueRange, 0.0, 1.0);
        setColor(context, 0.25, 0.62, 0.86);
        cairo_rectangle(context,
                        plotX + category * barWidth + barWidth * 0.15,
                        plotY + plotHeight - barHeight,
                        barWidth * 0.7,
                        barHeight);
        cairo_fill(context);

        char label[32];
        if (percentage)
            std::snprintf(label, sizeof(label), "%.0f%%", values[category] * 100.0);
        else
            std::snprintf(label, sizeof(label), "%.0f", values[category]);
        setColor(context, 0.82, 0.84, 0.88);
        drawText(context, plotX + category * barWidth + barWidth * 0.25, plotY + plotHeight + 17.0, label, 10.0);
        std::snprintf(label, sizeof(label), "%zu", category);
        drawText(context, plotX + category * barWidth + barWidth * 0.4, y + height - 4.0, label, 10.0);
    }
}

void drawLossChart(
    cairo_t *context,
    double x,
    double y,
    double width,
    double height,
    const std::vector<double> &trainingLosses,
    const std::vector<double> &testLosses)
{
    drawChartFrame(context, x, y, width, height, "Training and test loss");
    if (trainingLosses.empty())
        return;

    double minimum = std::min(*std::min_element(trainingLosses.begin(), trainingLosses.end()),
                              *std::min_element(testLosses.begin(), testLosses.end()));
    double maximum = std::max(*std::max_element(trainingLosses.begin(), trainingLosses.end()),
                              *std::max_element(testLosses.begin(), testLosses.end()));
    const double padding = std::max((maximum - minimum) * 0.1, 1e-6);
    minimum -= padding;
    maximum += padding;

    drawLineSeries(context, x, y, width, height, trainingLosses, minimum, maximum, 0.25, 0.72, 0.42);
    drawLineSeries(context, x, y, width, height, testLosses, minimum, maximum, 0.95, 0.55, 0.25);

    setColor(context, 0.25, 0.72, 0.42);
    drawText(context, x + margin, y + height - 5.0, "Training");
    setColor(context, 0.95, 0.55, 0.25);
    drawText(context, x + margin + 90.0, y + height - 5.0, "Test");
}

void drawActivationHistoryChart(
    cairo_t *context,
    double x,
    double y,
    double width,
    double height,
    const std::vector<Metrics::ActivationHistoryEntry> &history,
    size_t layerIndex)
{
    char title[80];
    std::snprintf(title, sizeof(title), "Hidden layer %zu activation saturation and sigmoid derivative",
                  layerIndex + 1);
    drawChartFrame(context, x, y, width, height, title);
    if (history.empty())
        return;

    std::vector<double> saturatedFractions;
    std::vector<double> normalizedMeanDerivatives;
    std::array<std::vector<double>, Metrics::ACTIVATION_DERIVATIVE_BIN_COUNT> derivativeBinFractions;
    saturatedFractions.reserve(history.size());
    normalizedMeanDerivatives.reserve(history.size());
    for (std::vector<double> &fractions : derivativeBinFractions)
        fractions.reserve(history.size());
    for (const Metrics::ActivationHistoryEntry &entry : history)
    {
        const Metrics::ActivationLayerMetrics &layer = entry.diagnostics.layers[layerIndex];
        saturatedFractions.push_back(layer.saturatedFraction);
        normalizedMeanDerivatives.push_back(layer.meanSigmoidDerivative / 0.25);
        for (size_t bin = 0; bin < derivativeBinFractions.size(); ++bin)
            derivativeBinFractions[bin].push_back(layer.derivativeBinFractions[bin]);
    }

    drawLineSeries(context, x, y, width, height, saturatedFractions,
                   0.0, 1.0, 0.95, 0.55, 0.25);
    drawLineSeries(context, x, y, width, height, normalizedMeanDerivatives,
                   0.0, 1.0, 0.25, 0.72, 0.42);
    const std::array<std::array<double, 3>, Metrics::ACTIVATION_DERIVATIVE_BIN_COUNT> binColors{{
        {0.92, 0.35, 0.34},
        {0.93, 0.68, 0.28},
        {0.45, 0.72, 0.90},
        {0.72, 0.55, 0.86},
        {0.42, 0.82, 0.76}}};
    for (size_t bin = 0; bin < derivativeBinFractions.size(); ++bin)
    {
        drawLineSeries(
            context,
            x,
            y,
            width,
            height,
            derivativeBinFractions[bin],
            0.0,
            1.0,
            binColors[bin][0],
            binColors[bin][1],
            binColors[bin][2]);
    }

    const Metrics::ActivationLayerMetrics &latest = history.back().diagnostics.layers[layerIndex];
    char summary[512];
    std::snprintf(summary, sizeof(summary),
                  "Steps %zu-%zu | saturated: %.1f%% | mean sigma'(Z): %.5f | bins <.01/.01-.05/.05-.1/.1-.2/>=.2: %.1f/%.1f/%.1f/%.1f/%.1f%%",
                  history.front().step,
                  history.back().step,
                  latest.saturatedFraction * 100.0,
                  latest.meanSigmoidDerivative,
                  latest.derivativeBinFractions[0] * 100.0,
                  latest.derivativeBinFractions[1] * 100.0,
                  latest.derivativeBinFractions[2] * 100.0,
                  latest.derivativeBinFractions[3] * 100.0,
                  latest.derivativeBinFractions[4] * 100.0);
    setColor(context, 0.88, 0.90, 0.94);
    drawText(context, x + margin, y + height - 24.0, summary, 11.0);
    setColor(context, 0.95, 0.55, 0.25);
    drawText(context, x + margin, y + height - 5.0, "Saturated");
    setColor(context, 0.25, 0.72, 0.42);
    drawText(context, x + margin + 80.0, y + height - 5.0, "Mean derivative / 0.25");
    const std::array<const char *, Metrics::ACTIVATION_DERIVATIVE_BIN_COUNT> binLabels{
        "d<.01", ".01-.05", ".05-.10", ".10-.20", "d>=.20"};
    double legendX = x + margin + 285.0;
    for (size_t bin = 0; bin < binLabels.size(); ++bin)
    {
        setColor(context, binColors[bin][0], binColors[bin][1], binColors[bin][2]);
        drawText(context, legendX, y + height - 5.0, binLabels[bin]);
        legendX += 75.0;
    }
}

void drawConfusionMatrix(
    cairo_t *context,
    double x,
    double y,
    double width,
    double height,
    const std::array<double, 100> &values)
{
    drawChartFrame(context, x, y, width, height, "Row-normalized confusion matrix");
    const double cell = std::max(1.0, std::min((width - margin - 24.0) / 10.0, (height - titleHeight - 40.0) / 10.0));
    const double startX = x + margin;
    const double startY = y + titleHeight + 4.0;

    for (size_t actual = 0; actual < 10; ++actual)
    {
        for (size_t predicted = 0; predicted < 10; ++predicted)
        {
            const double value = std::clamp(values[actual * 10 + predicted], 0.0, 1.0);
            setColor(context, 0.10 + 0.25 * value, 0.20 + 0.60 * value, 0.32 + 0.62 * value);
            cairo_rectangle(context, startX + predicted * cell, startY + actual * cell, cell - 1.0, cell - 1.0);
            cairo_fill(context);
            if (cell >= 28.0)
            {
                char label[16];
                std::snprintf(label, sizeof(label), "%.0f%%", value * 100.0);
                setColor(context, 0.96, 0.97, 0.99);
                drawText(context, startX + predicted * cell + 3.0, startY + actual * cell + cell * 0.62, label, 9.0);
            }
        }
    }
}

void drawCountsTable(
    cairo_t *context,
    double x,
    double y,
    double width,
    const std::array<size_t, 10> &actualCounts,
    const std::array<double, 10> &predictedCounts,
    const std::array<size_t, 100> &confusionCounts)
{
    setColor(context, 0.14, 0.16, 0.20);
    cairo_rectangle(context, x, y, width, 250.0);
    cairo_fill(context);
    setColor(context, 0.88, 0.90, 0.94);
    drawText(context, x + 14.0, y + 22.0, "Class counts", 15.0);

    const char *headers[] = {"Class", "Actual", "Predicted", "Correct"};
    const double columns[] = {x + 24.0, x + width * 0.34, x + width * 0.56, x + width * 0.80};
    setColor(context, 0.64, 0.69, 0.76);
    for (size_t column = 0; column < 4; ++column)
        drawText(context, columns[column], y + 50.0, headers[column]);

    for (size_t category = 0; category < actualCounts.size(); ++category)
    {
        char values[4][32];
        std::snprintf(values[0], sizeof(values[0]), "%zu", category);
        std::snprintf(values[1], sizeof(values[1]), "%zu", actualCounts[category]);
        std::snprintf(values[2], sizeof(values[2]), "%.0f", predictedCounts[category]);
        std::snprintf(values[3], sizeof(values[3]), "%zu", confusionCounts[category * 10 + category]);
        setColor(context, 0.86, 0.88, 0.92);
        for (size_t column = 0; column < 4; ++column)
            drawText(context, columns[column], y + 72.0 + category * 17.0, values[column]);
    }
}
}

TrainingMetricsGui::TrainingMetricsGui()
{
    gtk_init();
    mainLoop = g_main_loop_new(nullptr, FALSE);
    window = GTK_WINDOW(gtk_window_new());
    if (window == nullptr)
    {
        g_main_loop_unref(mainLoop);
        mainLoop = nullptr;
        throw std::runtime_error("Could not create GTK training metrics window");
    }

    g_object_ref_sink(window);
    gtk_window_set_title(window, "NN Training Metrics");
    gtk_window_set_default_size(window, 1100, 760);
    g_signal_connect(window, "close-request", G_CALLBACK(onCloseRequest), this);

    GtkWidget *root = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    GtkWidget *controls = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_widget_set_margin_top(controls, 8);
    gtk_widget_set_margin_start(controls, 12);
    gtk_widget_set_margin_end(controls, 12);
    gtk_widget_set_margin_bottom(controls, 4);

    startButton = gtk_button_new_with_label("Start");
    pauseButton = gtk_button_new_with_label("Pause");
    stopButton = gtk_button_new_with_label("Stop");
    statusLabel = gtk_label_new("Bereit");
    gtk_widget_set_hexpand(statusLabel, TRUE);
    gtk_label_set_xalign(GTK_LABEL(statusLabel), 1.0f);
    gtk_box_append(GTK_BOX(controls), startButton);
    gtk_box_append(GTK_BOX(controls), pauseButton);
    gtk_box_append(GTK_BOX(controls), stopButton);
    gtk_box_append(GTK_BOX(controls), statusLabel);
    g_signal_connect(startButton, "clicked", G_CALLBACK(onStartClicked), this);
    g_signal_connect(pauseButton, "clicked", G_CALLBACK(onPauseClicked), this);
    g_signal_connect(stopButton, "clicked", G_CALLBACK(onStopClicked), this);
    gtk_box_append(GTK_BOX(root), controls);

    GtkWidget *imageControls = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_widget_set_margin_start(imageControls, 12);
    gtk_widget_set_margin_end(imageControls, 12);
    gtk_widget_set_margin_bottom(imageControls, 8);
    GtkWidget *imageIndexLabel = gtk_label_new("Bildindex (0-basiert):");
    imageIndexEntry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(imageIndexEntry), "z. B. 1200");
    gtk_entry_set_input_purpose(GTK_ENTRY(imageIndexEntry), GTK_INPUT_PURPOSE_DIGITS);
    gtk_widget_set_size_request(imageIndexEntry, 120, -1);
    GtkWidget *imageLookupButton = gtk_button_new_with_label("Bild anzeigen");
    imageLookupStatus = gtk_label_new("Index und Label erscheinen hier.");
    gtk_label_set_xalign(GTK_LABEL(imageLookupStatus), 0.0f);
    gtk_widget_set_hexpand(imageLookupStatus, TRUE);
    imagePreviewArea = gtk_drawing_area_new();
    gtk_drawing_area_set_content_width(GTK_DRAWING_AREA(imagePreviewArea), 180);
    gtk_drawing_area_set_content_height(GTK_DRAWING_AREA(imagePreviewArea), 140);
    gtk_drawing_area_set_draw_func(
        GTK_DRAWING_AREA(imagePreviewArea), imagePreviewDrawCallback, this, nullptr);
    g_signal_connect(imageLookupButton, "clicked", G_CALLBACK(onImageLookupClicked), this);
    gtk_box_append(GTK_BOX(imageControls), imageIndexLabel);
    gtk_box_append(GTK_BOX(imageControls), imageIndexEntry);
    gtk_box_append(GTK_BOX(imageControls), imageLookupButton);
    gtk_box_append(GTK_BOX(imageControls), imageLookupStatus);
    gtk_box_append(GTK_BOX(imageControls), imagePreviewArea);
    gtk_box_append(GTK_BOX(root), imageControls);

    GtkWidget *scrolledWindow = gtk_scrolled_window_new();
    drawingArea = gtk_drawing_area_new();
    gtk_drawing_area_set_content_width(GTK_DRAWING_AREA(drawingArea), 1040);
    gtk_drawing_area_set_content_height(GTK_DRAWING_AREA(drawingArea), 1960);
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(drawingArea), drawCallback, this, nullptr);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scrolledWindow), drawingArea);
    gtk_widget_set_vexpand(scrolledWindow, TRUE);
    gtk_box_append(GTK_BOX(root), scrolledWindow);
    gtk_window_set_child(window, root);
    updateControls();
    gtk_window_present(window);
    uiUpdateSourceId = g_timeout_add(50, onUiUpdate, this);
}

TrainingMetricsGui::~TrainingMetricsGui()
{
    stopTraining();
    if (trainingThread.joinable())
        trainingThread.join();
    if (uiUpdateSourceId != 0)
        g_source_remove(uiUpdateSourceId);
    if (window != nullptr)
    {
        gtk_window_destroy(window);
        g_object_unref(window);
    }
    if (mainLoop != nullptr)
        g_main_loop_unref(mainLoop);
}

gboolean TrainingMetricsGui::onCloseRequest(GtkWindow *, gpointer userData)
{
    auto *gui = static_cast<TrainingMetricsGui *>(userData);
    gui->stopTraining();
    g_main_loop_quit(gui->mainLoop);
    return TRUE;
}

void TrainingMetricsGui::onStartClicked(GtkButton *, gpointer userData)
{
    static_cast<TrainingMetricsGui *>(userData)->startTraining();
}

void TrainingMetricsGui::onPauseClicked(GtkButton *, gpointer userData)
{
    static_cast<TrainingMetricsGui *>(userData)->pauseTraining();
}

void TrainingMetricsGui::onStopClicked(GtkButton *, gpointer userData)
{
    static_cast<TrainingMetricsGui *>(userData)->stopTraining();
}

void TrainingMetricsGui::onImageLookupClicked(GtkButton *, gpointer userData)
{
    static_cast<TrainingMetricsGui *>(userData)->showSelectedImage();
}

gboolean TrainingMetricsGui::onUiUpdate(gpointer userData)
{
    TrainingMetricsGui *gui = static_cast<TrainingMetricsGui *>(userData);
    gui->applyQueuedMetrics();
    if (gui->workerFinished.load(std::memory_order_acquire))
    {
        if (gui->trainingThread.joinable())
            gui->trainingThread.join();
        if (gui->trainingError)
        {
            gui->trainingActive = false;
            gui->trainingStopped = true;
            gtk_label_set_text(GTK_LABEL(gui->statusLabel), "Fehler beim Training");
            gui->updateControls();
            g_main_loop_quit(gui->mainLoop);
        }
        else if (!gui->trainingStopped)
        {
            gui->trainingActive = false;
            gui->trainingCompleted = true;
            gtk_label_set_text(GTK_LABEL(gui->statusLabel), "Abgeschlossen");
            gui->updateControls();
        }
    }

    return G_SOURCE_CONTINUE;
}

void TrainingMetricsGui::drawCallback(
    GtkDrawingArea *,
    cairo_t *context,
    int width,
    int height,
    gpointer userData)
{
    static_cast<TrainingMetricsGui *>(userData)->draw(context, width, height);
}

void TrainingMetricsGui::setTrainingStepCallback(TrainingStepCallback callback, void *userData)
{
    if (callback == nullptr)
        throw std::invalid_argument("Training step callback must not be empty");
    trainingStepCallback = callback;
    trainingStepUserData = userData;
}

void TrainingMetricsGui::setImageData(
    const float *images,
    const float *labels,
    size_t imageCount,
    size_t imageWidth,
    size_t imageHeight)
{
    if (images == nullptr || labels == nullptr || imageCount == 0 ||
        imageWidth == 0 || imageHeight == 0)
    {
        throw std::invalid_argument("Image preview requires images, labels and positive dimensions");
    }
    if (imageWidth > std::numeric_limits<size_t>::max() / imageHeight)
        throw std::length_error("Image preview dimensions overflow");

    const size_t imageStride = imageWidth * imageHeight;
    if (imageCount > std::numeric_limits<size_t>::max() / imageStride)
        throw std::length_error("Image preview data size overflows");

    imageData = images;
    imageLabels = labels;
    this->imageCount = imageCount;
    this->imageWidth = imageWidth;
    this->imageHeight = imageHeight;
}

void TrainingMetricsGui::run()
{
    if (trainingStepCallback == nullptr)
        throw std::logic_error("Training step callback has not been set");
    g_main_loop_run(mainLoop);
    if (trainingThread.joinable())
        trainingThread.join();
    rethrowTrainingError();
}

void TrainingMetricsGui::rethrowTrainingError() const
{
    if (trainingError)
        std::rethrow_exception(trainingError);
}

void TrainingMetricsGui::startTraining()
{
    if (trainingStopped || trainingCompleted || trainingStepCallback == nullptr)
        return;

    trainingStarted = true;
    trainingActive = true;
    {
        std::lock_guard<std::mutex> lock(trainingMutex);
        workerActive = true;
    }
    if (!trainingThread.joinable())
    {
        workerFinished.store(false, std::memory_order_release);
        trainingThread = std::thread(&TrainingMetricsGui::runTrainingWorker, this);
    }
    trainingCondition.notify_one();
    gtk_label_set_text(GTK_LABEL(statusLabel), "Training laeuft");
    updateControls();
}

void TrainingMetricsGui::pauseTraining()
{
    if (!trainingActive || trainingStopped || trainingCompleted)
        return;

    trainingActive = false;
    {
        std::lock_guard<std::mutex> lock(trainingMutex);
        workerActive = false;
    }
    gtk_label_set_text(GTK_LABEL(statusLabel), "Pausiert");
    updateControls();
}

void TrainingMetricsGui::stopTraining()
{
    trainingActive = false;
    trainingStopped = true;
    {
        std::lock_guard<std::mutex> lock(trainingMutex);
        workerActive = false;
        workerStopped = true;
    }
    trainingCondition.notify_one();
    if (statusLabel != nullptr)
        gtk_label_set_text(GTK_LABEL(statusLabel), "Gestoppt");
    updateControls();
}

void TrainingMetricsGui::showSelectedImage()
{
    const char *entryText = gtk_editable_get_text(GTK_EDITABLE(imageIndexEntry));
    char *endPointer = nullptr;
    errno = 0;
    const unsigned long long parsedIndex = g_ascii_strtoull(entryText, &endPointer, 10);
    if (entryText[0] == '\0' || endPointer == entryText ||
        *endPointer != '\0' || errno == ERANGE)
    {
        gtk_label_set_text(GTK_LABEL(imageLookupStatus), "Bitte einen gültigen Index eingeben.");
        return;
    }
    if (imageData == nullptr || imageLabels == nullptr || imageCount == 0)
    {
        gtk_label_set_text(GTK_LABEL(imageLookupStatus), "Bilddaten sind nicht verfügbar.");
        return;
    }
    if (parsedIndex >= imageCount)
    {
        char message[128];
        std::snprintf(
            message,
            sizeof(message),
            "Index außerhalb des Bereichs 0 bis %zu.",
            imageCount - 1);
        gtk_label_set_text(GTK_LABEL(imageLookupStatus), message);
        return;
    }

    selectedImageIndex = static_cast<size_t>(parsedIndex);
    hasSelectedImage = true;
    char message[128];
    std::snprintf(
        message,
        sizeof(message),
        "Ausgewählt: Bild %zu, Label %.0f",
        selectedImageIndex,
        imageLabels[selectedImageIndex]);
    gtk_label_set_text(GTK_LABEL(imageLookupStatus), message);
    gtk_widget_queue_draw(imagePreviewArea);
}

void TrainingMetricsGui::imagePreviewDrawCallback(
    GtkDrawingArea *,
    cairo_t *context,
    int width,
    int height,
    gpointer userData)
{
    static_cast<TrainingMetricsGui *>(userData)->drawImagePreview(context, width, height);
}

void TrainingMetricsGui::drawImagePreview(cairo_t *context, int width, int height) const
{
    setColor(context, 0.14, 0.16, 0.20);
    cairo_paint(context);
    if (!hasSelectedImage || imageData == nullptr)
    {
        setColor(context, 0.82, 0.84, 0.88);
        drawText(context, 8.0, static_cast<double>(height) / 2.0, "Kein Bild ausgewählt", 11.0);
        return;
    }

    constexpr double PREVIEW_PADDING = 8.0;
    constexpr double CAPTION_HEIGHT = 22.0;
    const double scale = std::min(
        (static_cast<double>(width) - PREVIEW_PADDING * 2.0) / static_cast<double>(imageWidth),
        (static_cast<double>(height) - CAPTION_HEIGHT - PREVIEW_PADDING * 2.0) /
            static_cast<double>(imageHeight));
    const double imageDrawWidth = static_cast<double>(imageWidth) * scale;
    const double imageDrawHeight = static_cast<double>(imageHeight) * scale;
    const double imageX = (static_cast<double>(width) - imageDrawWidth) / 2.0;
    const double imageY = PREVIEW_PADDING;

    for (size_t row = 0; row < imageHeight; ++row)
    {
        for (size_t column = 0; column < imageWidth; ++column)
        {
            const size_t feature = row * imageWidth + column;
            double value = static_cast<double>(
                imageData[feature * imageCount + selectedImageIndex]);
            if (!std::isfinite(value))
                value = 0.0;
            value = std::clamp(value, 0.0, 1.0);
            cairo_set_source_rgb(context, value, value, value);
            cairo_rectangle(
                context,
                imageX + static_cast<double>(column) * scale,
                imageY + static_cast<double>(row) * scale,
                scale + 0.1,
                scale + 0.1);
            cairo_fill(context);
        }
    }

    char caption[80];
    std::snprintf(
        caption,
        sizeof(caption),
        "Index %zu | Label %.0f",
        selectedImageIndex,
        imageLabels[selectedImageIndex]);
    setColor(context, 0.90, 0.91, 0.94);
    drawText(context, 8.0, static_cast<double>(height) - 6.0, caption, 11.0);
}

void TrainingMetricsGui::updateControls()
{
    if (startButton == nullptr)
        return;
    gtk_button_set_label(GTK_BUTTON(startButton), trainingStarted ? "Fortsetzen" : "Start");
    gtk_widget_set_sensitive(startButton, !trainingActive && !trainingStopped && !trainingCompleted);
    gtk_widget_set_sensitive(pauseButton, trainingActive);
    gtk_widget_set_sensitive(stopButton, !trainingStopped && !trainingCompleted);
}

void TrainingMetricsGui::queueMetrics(const Metrics::TrainingMetricsSnapshot &metrics)
{
    std::lock_guard<std::mutex> lock(trainingMutex);
    pendingMetrics.push_back(metrics);
}

void TrainingMetricsGui::runTrainingWorker()
{
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(trainingMutex);
            while (!workerActive && !workerStopped)
                trainingCondition.wait(lock);
            if (workerStopped)
                break;
        }

        try
        {
            if (!trainingStepCallback(trainingStepUserData))
            {
                std::lock_guard<std::mutex> lock(trainingMutex);
                workerActive = false;
                workerStopped = true;
            }
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(trainingMutex);
            trainingError = std::current_exception();
            workerActive = false;
            workerStopped = true;
        }

        {
            std::lock_guard<std::mutex> lock(trainingMutex);
            if (workerStopped)
                break;
        }
    }
    workerFinished.store(true, std::memory_order_release);
}

void TrainingMetricsGui::applyQueuedMetrics()
{
    std::vector<Metrics::TrainingMetricsSnapshot> metricsToApply;
    {
        std::lock_guard<std::mutex> lock(trainingMutex);
        metricsToApply.swap(pendingMetrics);
    }

    for (const Metrics::TrainingMetricsSnapshot &metrics : metricsToApply)
    {
        steps.push_back(static_cast<double>(metrics.step));
        trainingLosses.push_back(metrics.trainingLoss);
        testLosses.push_back(metrics.testLoss);
        testAccuracies.push_back(metrics.testAccuracy);
        actualCounts = metrics.actualCounts;
        confusionCounts = metrics.confusionCounts;
        normalizedConfusion = metrics.normalizedConfusion;
        meanConfidence = metrics.meanConfidence;
        activationHistory.push_back({metrics.step, metrics.activationDiagnostics});
        hasDiagnostics = true;
        for (size_t category = 0; category < predictedCounts.size(); ++category)
        {
            predictedCounts[category] = static_cast<double>(metrics.predictedCounts[category]);
            classRecall[category] = static_cast<double>(metrics.classRecall[category]);
        }
    }

    if (!metricsToApply.empty())
        render();
}

void TrainingMetricsGui::render()
{
    if (drawingArea == nullptr)
        return;
    gtk_widget_queue_draw(drawingArea);
}

void TrainingMetricsGui::draw(cairo_t *context, int width, int) const
{
    setColor(context, 0.08, 0.09, 0.12);
    cairo_paint(context);

    setColor(context, 0.94, 0.95, 0.98);
    drawText(context, 30.0, 36.0, "Training metrics", 22.0);

    char confidence[80];
    std::snprintf(confidence, sizeof(confidence), "Mean confidence: %.4f", meanConfidence);
    drawText(context, 30.0, 60.0, hasDiagnostics ? confidence : "Waiting for first evaluation.");

    const double gap = 20.0;
    const double chartWidth = std::max(300.0, (static_cast<double>(width) - 60.0 - gap) / 2.0);
    const double left = 30.0;
    const double right = left + chartWidth + gap;
    drawLossChart(context, left, 80.0, chartWidth, 250.0, trainingLosses, testLosses);

    double maximumPredictions = 0.0;
    for (double value : predictedCounts)
        maximumPredictions = std::max(maximumPredictions, value);
    drawBarChart(context, right, 80.0, chartWidth, 250.0, "Predicted class distribution",
                  predictedCounts, maximumPredictions, false);

    if (hasDiagnostics)
    {
        drawConfusionMatrix(context, left, 350.0, chartWidth, 360.0, normalizedConfusion);
        drawBarChart(context, right, 350.0, chartWidth, 360.0, "Recall by actual class",
                      classRecall, 1.0, true);
        drawCountsTable(context, left, 730.0, static_cast<double>(width) - 60.0,
                        actualCounts, predictedCounts, confusionCounts);
    }

    if (!testAccuracies.empty())
    {
        std::vector<double> accuracy = testAccuracies;
        drawChartFrame(context, left, 1000.0, static_cast<double>(width) - 60.0, 250.0, "Test accuracy");
        drawLineSeries(context, left, 1000.0, static_cast<double>(width) - 60.0, 250.0,
                       accuracy, 0.0, 1.0, 0.35, 0.68, 0.95);
    }

    for (size_t layer = 0; layer < 3; ++layer)
    {
        drawActivationHistoryChart(
            context,
            left,
            1280.0 + layer * 220.0,
            static_cast<double>(width) - 60.0,
            205.0,
            activationHistory,
            layer);
    }
}
