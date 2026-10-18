#include "TrainingMetricsGui.hpp"

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>
#include <implot.h>

#include <stdexcept>
#include <string>

TrainingMetricsGui::TrainingMetricsGui()
{
    for (size_t category = 0; category < classPositions.size(); ++category)
        classPositions[category] = static_cast<double>(category);

    if (!SDL_Init(SDL_INIT_VIDEO))
        throw std::runtime_error(SDL_GetError());
    sdlInitialized = true;

    window = SDL_CreateWindow("NN Training Metrics", 1100, 760, SDL_WINDOW_RESIZABLE);
    if (window == nullptr)
    {
        const std::string error = SDL_GetError();
        shutdown();
        throw std::runtime_error(error);
    }

    renderer = SDL_CreateRenderer(window, nullptr);
    if (renderer == nullptr)
    {
        const std::string error = SDL_GetError();
        shutdown();
        throw std::runtime_error(error);
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    imguiInitialized = true;
    ImGui::StyleColorsDark();

    if (!ImGui_ImplSDL3_InitForSDLRenderer(window, renderer))
    {
        shutdown();
        throw std::runtime_error("Dear ImGui SDL3 backend initialization failed");
    }
    imguiSdlBackendInitialized = true;

    if (!ImGui_ImplSDLRenderer3_Init(renderer))
    {
        shutdown();
        throw std::runtime_error("Dear ImGui SDL renderer backend initialization failed");
    }
    imguiRendererBackendInitialized = true;
}

TrainingMetricsGui::~TrainingMetricsGui()
{
    shutdown();
}

void TrainingMetricsGui::shutdown() noexcept
{
    if (imguiRendererBackendInitialized)
        ImGui_ImplSDLRenderer3_Shutdown();
    if (imguiSdlBackendInitialized)
        ImGui_ImplSDL3_Shutdown();
    if (imguiInitialized)
    {
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }
    if (renderer != nullptr)
        SDL_DestroyRenderer(renderer);
    if (window != nullptr)
        SDL_DestroyWindow(window);
    if (sdlInitialized)
        SDL_Quit();

    imguiRendererBackendInitialized = false;
    imguiSdlBackendInitialized = false;
    imguiInitialized = false;
    renderer = nullptr;
    window = nullptr;
    sdlInitialized = false;
}

bool TrainingMetricsGui::processEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        ImGui_ImplSDL3_ProcessEvent(&event);
        if (event.type == SDL_EVENT_QUIT)
            running = false;
        if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
            event.window.windowID == SDL_GetWindowID(window))
            running = false;
    }
    return running;
}

void TrainingMetricsGui::addMetrics(
    size_t step,
    float trainingLoss,
    float testLoss,
    float testAccuracy)
{
    steps.push_back(static_cast<double>(step));
    trainingLosses.push_back(trainingLoss);
    testLosses.push_back(testLoss);
    testAccuracies.push_back(testAccuracy);
}

void TrainingMetricsGui::setDiagnostics(
    const std::array<size_t, 10> &actualCounts,
    const std::array<size_t, 10> &predictedCounts,
    const std::array<size_t, 100> &confusionCounts,
    const std::array<double, 100> &normalizedConfusion,
    const std::array<float, 10> &classRecall,
    float meanConfidence)
{
    this->actualCounts = actualCounts;
    this->confusionCounts = confusionCounts;
    this->normalizedConfusion = normalizedConfusion;
    this->meanConfidence = meanConfidence;
    hasDiagnostics = true;
    for (size_t category = 0; category < classPositions.size(); ++category)
    {
        this->predictedCounts[category] = static_cast<double>(predictedCounts[category]);
        this->classRecall[category] = static_cast<double>(classRecall[category]);
    }
}

void TrainingMetricsGui::render()
{
    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowSize(ImVec2(1000.0f, 680.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Training metrics");
    ImGui::Text("Test metrics are measured every 100 training steps.");
    if (hasDiagnostics)
        ImGui::Text("Mean confidence: %.4f", meanConfidence);

    if (!steps.empty())
    {
        const int count = static_cast<int>(steps.size());
        if (ImPlot::BeginPlot("Loss"))
        {
            ImPlot::SetupAxes("Training step", "Cross-entropy");
            ImPlot::PlotLine("Training loss", steps.data(), trainingLosses.data(), count);
            ImPlot::PlotLine("Test loss", steps.data(), testLosses.data(), count);
            ImPlot::EndPlot();
        }

        if (hasDiagnostics && ImPlot::BeginPlot("Predicted class distribution"))
        {
            ImPlot::SetupAxes("Predicted class", "Number of test samples");
            ImPlot::SetupAxisTicks(ImAxis_X1, classPositions.data(), static_cast<int>(classPositions.size()));
            ImPlot::PlotBars("Predicted", classPositions.data(), predictedCounts.data(),
                             static_cast<int>(predictedCounts.size()), 0.7);
            ImPlot::EndPlot();
        }

        if (hasDiagnostics && ImPlot::BeginPlot("Confusion matrix (row-normalized)"))
        {
            constexpr const char *classLabels[] = {
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
            ImPlot::SetupAxes("Predicted class", "Actual class");
            ImPlot::SetupAxisTicks(ImAxis_X1, 0.5, 9.5, 10, classLabels, false);
            ImPlot::SetupAxisTicks(ImAxis_Y1, 0.5, 9.5, 10, classLabels, false);
            ImPlot::PlotHeatmap("Recall", normalizedConfusion.data(), 10, 10, 0.0, 1.0,
                                "%.2f", ImPlotPoint(0, 0), ImPlotPoint(10, 10));
            ImPlot::EndPlot();
        }

        if (hasDiagnostics && ImPlot::BeginPlot("Recall by actual class"))
        {
            ImPlot::SetupAxes("Actual class", "Recall");
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Always);
            ImPlot::SetupAxisTicks(ImAxis_X1, classPositions.data(), static_cast<int>(classPositions.size()));
            ImPlot::PlotBars("Recall", classPositions.data(), classRecall.data(),
                             static_cast<int>(classRecall.size()), 0.7);
            ImPlot::EndPlot();
        }

        if (hasDiagnostics && ImGui::CollapsingHeader("Class counts"))
        {
            if (ImGui::BeginTable("ClassCounts", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
            {
                ImGui::TableSetupColumn("Class");
                ImGui::TableSetupColumn("Actual");
                ImGui::TableSetupColumn("Predicted");
                ImGui::TableSetupColumn("Correct");
                ImGui::TableHeadersRow();
                for (size_t category = 0; category < actualCounts.size(); ++category)
                {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%zu", category);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%zu", actualCounts[category]);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.0f", predictedCounts[category]);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%zu", confusionCounts[category * 10 + category]);
                }
                ImGui::EndTable();
            }
        }

        if (ImPlot::BeginPlot("Test accuracy"))
        {
            ImPlot::SetupAxes("Training step", "Accuracy");
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Always);
            ImPlot::PlotLine("Test accuracy", steps.data(), testAccuracies.data(), count);
            ImPlot::EndPlot();
        }
    }
    else
    {
        ImGui::Text("Waiting for first evaluation.");
    }

    ImGui::End();
    ImGui::Render();

    if (!SDL_SetRenderDrawColor(renderer, 20, 22, 28, 255) || !SDL_RenderClear(renderer))
        throw std::runtime_error(SDL_GetError());
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
    SDL_RenderPresent(renderer);
}
