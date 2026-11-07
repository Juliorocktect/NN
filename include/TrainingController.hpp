#ifndef TRAINING_CONTROLLER_HPP
#define TRAINING_CONTROLLER_HPP

#include "TrainingConfig.hpp"
#include "TrainingDataset.hpp"

/**
 * @brief Coordinates network training, periodic evaluation, metrics output, and GUI updates.
 */
class TrainingController
{
public:
    /**
     * @brief Validates and starts a training run using the provided dataset and settings.
     *
     * @param dataset Loaded training and test images and labels.
     * @param config Training step counts, evaluation interval, and image dimensions.
     * @throws std::invalid_argument If the dataset or configuration is invalid.
     * @throws std::runtime_error If training, evaluation, or metrics output fails.
     */
    void run(TrainingDataset &dataset, const TrainingConfig &config) const;

private:
    /** @brief Runtime data kept alive while the GUI invokes the training callback. */
    struct TrainingState;

    /**
     * @brief Runs one step and performs scheduled evaluation and reporting.
     *
     * @param userData Pointer to the active TrainingState.
     * @return true to continue training, or false when the configured final step completes.
     */
    static bool runTrainingStep(void *userData);

    /**
     * @brief Predicts the digit represented by a single 28-by-28 image.
     *
     * @param userData Pointer to the single-image prediction network.
     * @param image Input pixels normalized to the range [0, 1].
     * @param predictedClass Receives the predicted digit.
     * @param confidence Receives the predicted class probability.
     */
    static void predictImage(
        void *userData,
        const float *image,
        size_t &predictedClass,
        float &confidence);
};

#endif
