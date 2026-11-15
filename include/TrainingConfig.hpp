#ifndef TRAINING_CONFIG_HPP
#define TRAINING_CONFIG_HPP

#include <cstddef>

/**
 * @brief Dataset dimensions and runtime settings for a training run.
 */
struct TrainingConfig {
  /** @brief Maximum number of training samples to load and use. */
  size_t trainingCount = 9000;
  /** @brief Maximum number of test samples to load and evaluate. */
  size_t testCount = 30000;
  /** @brief Number of training steps before training completes. */
  size_t trainingSteps = 50000;
  /** @brief Number of steps between periodic evaluation and metric updates. */
  size_t evaluationInterval = 500;
  /** @brief Number of initial training samples used by the final gradient
   * check. */
  size_t gradientCheckSampleCount = 16;
  /** @brief Width of each input image in pixels. */
  size_t imageWidth = 28;
  /** @brief Height of each input image in pixels. */
  size_t imageHeight = 28;
};

struct ConvolutionConfig {
  const size_t kernelRowSize = 4;
  const size_t kernelColSize = 4;
};

#endif
