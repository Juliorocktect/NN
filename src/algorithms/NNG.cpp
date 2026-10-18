#include "VNN.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

NNG::NNG()
{
    y = GMatrix(10, SIZE_TRAINING_DATA);
    w1 = GMatrix(480, 784);
    w1.initRandom();
    w2 = GMatrix(200, 480);
    w2.initRandom();
    w3 = GMatrix(180, 200);
    w3.initRandom();
    w4 = GMatrix(10, 180);
    w4.initRandom();
    A1 = GMatrix(SIZE_FIRST_LAYER, SIZE_TRAINING_DATA);
    A2 = GMatrix(SIZE_SECOND_LAYER, SIZE_TRAINING_DATA);
    A3 = GMatrix(SIZE_THIRD_LAYER, SIZE_TRAINING_DATA);
    Z1 = GMatrix(SIZE_FIRST_LAYER, SIZE_TRAINING_DATA);
    Z2 = GMatrix(SIZE_SECOND_LAYER, SIZE_TRAINING_DATA);
    Z3 = GMatrix(SIZE_THIRD_LAYER, SIZE_TRAINING_DATA);
    y_hat = GMatrix(10, SIZE_TRAINING_DATA);
    b1 = GVector(480);
    b2 = GVector(200);
    b3 = GVector(180);
    b4 = GVector(10);
    dE_dYHAT = GMatrix(10, SIZE_TRAINING_DATA);
    db4 = GVector(10);
    db3 = GVector(180);
    db2 = GVector(200);
    db1 = GVector(480);
    dYHAT_dZ3 = GMatrix(180, SIZE_TRAINING_DATA);
    dZ2 = GMatrix(200, SIZE_TRAINING_DATA);
    dW4 = GMatrix(10, 180);
    dW3 = GMatrix(180, 200);
    dW2 = GMatrix(200, 480);
    dW1 = GMatrix(480, 784);
    inputData = GMatrix(784, SIZE_TRAINING_DATA);
    labels = GVector(SIZE_TRAINING_DATA);
}
NNG::NNG(size_t sizeTrainingData)
{
    this->SIZE_TRAINING_DATA = sizeTrainingData;
    y = GMatrix(10, SIZE_TRAINING_DATA);
    w1 = GMatrix(480, 784);
    w1.initRandom();
    w2 = GMatrix(200, 480);
    w2.initRandom();
    w3 = GMatrix(180, 200);
    w3.initRandom();
    w4 = GMatrix(10, 180);
    w4.initRandom();
    A1 = GMatrix(SIZE_FIRST_LAYER, SIZE_TRAINING_DATA);
    A2 = GMatrix(SIZE_SECOND_LAYER, SIZE_TRAINING_DATA);
    A3 = GMatrix(SIZE_THIRD_LAYER, SIZE_TRAINING_DATA);
    Z1 = GMatrix(SIZE_FIRST_LAYER, SIZE_TRAINING_DATA);
    Z2 = GMatrix(SIZE_SECOND_LAYER, SIZE_TRAINING_DATA);
    Z3 = GMatrix(SIZE_THIRD_LAYER, SIZE_TRAINING_DATA);
    y_hat = GMatrix(10, SIZE_TRAINING_DATA);
    b1 = GVector(480);
    b2 = GVector(200);
    b3 = GVector(180);
    b4 = GVector(10);
    dE_dYHAT = GMatrix(10, SIZE_TRAINING_DATA);
    db4 = GVector(10);
    db3 = GVector(180);
    db2 = GVector(200);
    db1 = GVector(480);
    dYHAT_dZ3 = GMatrix(180, SIZE_TRAINING_DATA);
    dZ2 = GMatrix(200, SIZE_TRAINING_DATA);
    dW4 = GMatrix(10, 180);
    dW3 = GMatrix(180, 200);
    dW2 = GMatrix(200, 480);
    dW1 = GMatrix(480, 784);
    inputData = GMatrix(784, SIZE_TRAINING_DATA);
    labels = GVector(SIZE_TRAINING_DATA);
}
NNG::~NNG()
{
}
void NNG::setInputData(const float *mat)
{
    inputData.uploadMatrixToGPU(mat);
}
void NNG::printGreen(const char *text)
{
    std::cout << "\033[1;32m" << text << "\033[0m\n";
}
void NNG::printRed(const char *text)
{
    std::cout << "\033[1;33m" << text << "\033[0m\n";
}
void NNG::initilizeYMatrix(float *pLabels)
{
    this->labels.upload(pLabels);
    // y muss mit 0 initialisiert sein!
    CudaLaunchers::hotEncodeYMatrix(labels.getVector(), y.getMatrix(), SIZE_TRAINING_DATA);
}
void NNG::setLabels(float *labels)
{
    this->labels.upload(labels);
}

float NNG::sumCrossEntropyLoss()
{
    return CudaLaunchers::sumCrossEntropyLoss(y_hat.getMatrix(), labels.getVector(), SIZE_TRAINING_DATA);
}
void NNG::run(size_t times)
{
    for (size_t i = 0; i < times; i++)
    {
        trainStep();
    }
}
float NNG::trainStep()
{
    forwardProp();
    const float loss = sumCrossEntropyLoss();
    backpropagateOutputLayer();
    backpropagateThirdLayer();
    backpropagateSecondLayer();
    backpropagateFirstLayer();
    updateWeightsAndBiases();
    return loss;
}

void NNG::copyParametersFrom(const NNG &source)
{
    w1 = source.w1;
    w2 = source.w2;
    w3 = source.w3;
    w4 = source.w4;
    b1 = source.b1;
    b2 = source.b2;
    b3 = source.b3;
    b4 = source.b4;
}

EvaluationMetrics NNG::evaluate(const float *testData, const float *testLabels)
{
    if (SIZE_TRAINING_DATA == 0 || testData == nullptr || testLabels == nullptr)
        throw std::invalid_argument("Evaluation requires test data and labels");

    setInputData(testData);
    forwardProp();

    const size_t sampleCount = SIZE_TRAINING_DATA;
    std::vector<float> predictions(10 * sampleCount);
    cudaError_t error = cudaMemcpy(
        predictions.data(),
        y_hat.getMatrix(),
        predictions.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    constexpr float minimumProbability = 1e-8f;
    EvaluationMetrics metrics;
    size_t correctCount = 0;
    float totalLoss = 0.0f;
    float totalConfidence = 0.0f;
    for (size_t sample = 0; sample < sampleCount; ++sample)
    {
        if (!std::isfinite(testLabels[sample]))
            throw std::invalid_argument("Test labels must be finite integers from 0 to 9");
        const int label = static_cast<int>(testLabels[sample]);
        if (label < 0 || label >= 10 || testLabels[sample] != label)
            throw std::invalid_argument("Test labels must be integers from 0 to 9");

        int predictedLabel = 0;
        float maximumProbability = predictions[sample];
        for (int category = 1; category < 10; ++category)
        {
            const float probability = predictions[category * sampleCount + sample];
            if (probability > maximumProbability)
            {
                maximumProbability = probability;
                predictedLabel = category;
            }
        }

        ++metrics.actualCounts[label];
        ++metrics.predictedCounts[predictedLabel];
        ++metrics.confusionCounts[label * 10 + predictedLabel];
        if (predictedLabel == label)
        {
            ++correctCount;
            ++metrics.classRecall[label];
        }
        totalConfidence += maximumProbability;
        const float probability = predictions[label * sampleCount + sample];
        totalLoss -= std::log(std::max(probability, minimumProbability));
    }

    metrics.loss = totalLoss / static_cast<float>(sampleCount);
    metrics.accuracy = static_cast<float>(correctCount) / static_cast<float>(sampleCount);
    metrics.meanConfidence = totalConfidence / static_cast<float>(sampleCount);

    for (size_t actual = 0; actual < metrics.actualCounts.size(); ++actual)
    {
        if (metrics.actualCounts[actual] == 0)
            continue;
        for (size_t predicted = 0; predicted < metrics.predictedCounts.size(); ++predicted)
        {
            const size_t index = actual * 10 + predicted;
            metrics.normalizedConfusion[index] =
                static_cast<double>(metrics.confusionCounts[index]) /
                static_cast<double>(metrics.actualCounts[actual]);
        }
        metrics.classRecall[actual] /= static_cast<float>(metrics.actualCounts[actual]);
    }
    return metrics;
}
float NNG::calcAccuracy(float *testData, float *testLabels, size_t sizeTestData)
{
    NNG evaluator(sizeTestData);
    evaluator.copyParametersFrom(*this);
    return evaluator.evaluate(testData, testLabels).accuracy;
}