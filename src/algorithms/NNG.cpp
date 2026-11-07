#include "VNN.h"
#include "CudaLaunchers.cuh"
#include <algorithm>
#include <limits>
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

void NNG::copyHiddenLayerActivations(std::array<GVector, 3> &activations)
{
    const std::array<GMatrix *, 3> layerActivations{&A1, &A2, &A3};
    for (size_t layer = 0; layer < layerActivations.size(); ++layer)
    {
        GMatrix *activation = layerActivations[layer];
        const size_t valueCount = activation->getRows() * activation->getCols();
        activations[layer].setSize(valueCount);
        if (valueCount == 0)
            continue;

        const cudaError_t error = cudaMemcpy(
            activations[layer].getVector(),
            activation->getMatrix(),
            valueCount * sizeof(float),
            cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }
}

double NNG::calculateGradientCheckLoss(const float *hostLabels, size_t sampleCount)
{
    forwardProp();

    std::vector<float> predictions(10 * sampleCount);
    const cudaError_t error = cudaMemcpy(
        predictions.data(),
        y_hat.getMatrix(),
        predictions.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    double totalLoss = 0.0;
    for (size_t sample = 0; sample < sampleCount; ++sample)
    {
        const size_t label = static_cast<size_t>(hostLabels[sample]);
        totalLoss -= std::log(std::max(
            static_cast<double>(predictions[label * sampleCount + sample]),
            1e-8));
    }
    return totalLoss / static_cast<double>(sampleCount);
}

bool NNG::checkGradientParameterBlock(
    const char *name,
    float *parameters,
    float *gradients,
    size_t parameterCount,
    const float *hostLabels,
    size_t sampleCount,
    float epsilon,
    float tolerance)
{
    if (parameterCount == 0)
        throw std::invalid_argument("Gradient check parameter block cannot be empty");

    std::vector<float> hostGradients(parameterCount);
    cudaError_t error = cudaMemcpy(
        hostGradients.data(),
        gradients,
        parameterCount * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    size_t index = 0;
    for (size_t current = 1; current < parameterCount; ++current)
    {
        if (std::abs(hostGradients[current]) > std::abs(hostGradients[index]))
            index = current;
    }

    const float analyticGradient = hostGradients[index];
    float originalValue = 0.0f;
    error = cudaMemcpy(
        &originalValue,
        parameters + index,
        sizeof(float),
        cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    const float delta = epsilon * std::max(1.0f, std::abs(originalValue));
    float perturbedValue = originalValue + delta;
    error = cudaMemcpy(
        parameters + index,
        &perturbedValue,
        sizeof(float),
        cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    double positiveLoss = 0.0;
    double negativeLoss = 0.0;
    try
    {
        positiveLoss = calculateGradientCheckLoss(hostLabels, sampleCount);

        perturbedValue = originalValue - delta;
        error = cudaMemcpy(
            parameters + index,
            &perturbedValue,
            sizeof(float),
            cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        negativeLoss = calculateGradientCheckLoss(hostLabels, sampleCount);
    }
    catch (...)
    {
        const cudaError_t restoreError = cudaMemcpy(
            parameters + index,
            &originalValue,
            sizeof(float),
            cudaMemcpyHostToDevice);
        if (restoreError != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(restoreError));
        throw;
    }

    error = cudaMemcpy(
        parameters + index,
        &originalValue,
        sizeof(float),
        cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    const double numericalGradient = (positiveLoss - negativeLoss) / (2.0 * delta);
    const double absoluteError = std::abs(numericalGradient - analyticGradient);
    const double relativeError = absoluteError /
                                 std::max(1e-6, std::abs(numericalGradient) + std::abs(analyticGradient));
    const bool passed = relativeError <= tolerance || absoluteError <= 1e-4;

    std::cout << "Gradient check " << name
              << "[" << index << "]"
              << " | analytisch=" << analyticGradient
              << " numerisch=" << numericalGradient
              << " relativer Fehler=" << relativeError
              << " | " << (passed ? "OK" : "FEHLER") << '\n';
    return passed;
}

bool NNG::runGradientCheck(
    const float *hostInputData,
    const float *hostLabels,
    size_t sampleCount,
    float epsilon,
    float tolerance) const
{
    if (hostInputData == nullptr || hostLabels == nullptr || sampleCount == 0)
        throw std::invalid_argument("Gradient check requires input data and labels");
    if (!std::isfinite(epsilon) || epsilon <= 0.0f ||
        !std::isfinite(tolerance) || tolerance <= 0.0f)
        throw std::invalid_argument("Gradient check epsilon and tolerance must be positive and finite");

    std::vector<float> labelCopy(hostLabels, hostLabels + sampleCount);
    for (float label : labelCopy)
    {
        if (!std::isfinite(label) || label < 0.0f || label >= 10.0f || label != std::floor(label))
            throw std::invalid_argument("Gradient check labels must be integers from 0 to 9");
    }

    NNG checker(sampleCount);
    checker.copyParametersFrom(*this);
    checker.setInputData(hostInputData);
    checker.initilizeYMatrix(labelCopy.data());

    checker.forwardProp();
    checker.backpropagateOutputLayer();
    checker.backpropagateThirdLayer();
    checker.backpropagateSecondLayer();
    checker.backpropagateFirstLayer();

    bool passed = true;
    passed &= checker.checkGradientParameterBlock(
        "W1", checker.w1.getMatrix(), checker.dW1.getMatrix(),
        checker.w1.getRows() * checker.w1.getCols(), labelCopy.data(), sampleCount, epsilon, tolerance);
    passed &= checker.checkGradientParameterBlock(
        "b1", checker.b1.getVector(), checker.db1.getVector(),
        checker.b1.getSize(), labelCopy.data(), sampleCount, epsilon, tolerance);
    passed &= checker.checkGradientParameterBlock(
        "W2", checker.w2.getMatrix(), checker.dW2.getMatrix(),
        checker.w2.getRows() * checker.w2.getCols(), labelCopy.data(), sampleCount, epsilon, tolerance);
    passed &= checker.checkGradientParameterBlock(
        "b2", checker.b2.getVector(), checker.db2.getVector(),
        checker.b2.getSize(), labelCopy.data(), sampleCount, epsilon, tolerance);
    passed &= checker.checkGradientParameterBlock(
        "W3", checker.w3.getMatrix(), checker.dW3.getMatrix(),
        checker.w3.getRows() * checker.w3.getCols(), labelCopy.data(), sampleCount, epsilon, tolerance);
    passed &= checker.checkGradientParameterBlock(
        "b3", checker.b3.getVector(), checker.db3.getVector(),
        checker.b3.getSize(), labelCopy.data(), sampleCount, epsilon, tolerance);
    passed &= checker.checkGradientParameterBlock(
        "W4", checker.w4.getMatrix(), checker.dW4.getMatrix(),
        checker.w4.getRows() * checker.w4.getCols(), labelCopy.data(), sampleCount, epsilon, tolerance);
    passed &= checker.checkGradientParameterBlock(
        "b4", checker.b4.getVector(), checker.db4.getVector(),
        checker.b4.getSize(), labelCopy.data(), sampleCount, epsilon, tolerance);

    std::cout << "Gradient check insgesamt: " << (passed ? "BESTANDEN" : "FEHLGESCHLAGEN") << '\n';
    return passed;
}

PredictionBatch NNG::predict(const float *data)
{
    if (SIZE_TRAINING_DATA == 0 || data == nullptr)
    {
        throw std::invalid_argument("Prediction requires a non-empty input batch");
    }

    const size_t sampleCount = SIZE_TRAINING_DATA;
    if (sampleCount > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("Prediction sample count exceeds CUDA launcher limits");
    }

    setInputData(data);
    forwardProp();

    GVector devicePredictions(sampleCount);
    CudaLaunchers::argmax(
        y_hat.getMatrix(),
        10,
        static_cast<int>(sampleCount),
        devicePredictions.getVector());

    std::vector<float> classProbabilities(10 * sampleCount);
    cudaError_t error = cudaMemcpy(
        classProbabilities.data(),
        y_hat.getMatrix(),
        classProbabilities.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    std::vector<float> hostPredictedClasses(sampleCount);
    error = cudaMemcpy(
        hostPredictedClasses.data(),
        devicePredictions.getVector(),
        sampleCount * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    PredictionBatch result;
    result.predictedClasses.resize(sampleCount);
    result.confidenceValues.resize(sampleCount);
    for (size_t sample = 0; sample < sampleCount; ++sample)
    {
        const size_t predictedClass = static_cast<size_t>(hostPredictedClasses[sample]);
        result.predictedClasses[sample] = predictedClass;
        result.confidenceValues[sample] =
            classProbabilities[predictedClass * sampleCount + sample];
    }
    return result;
}

float NNG::calculateEvaluationLoss(const float *hostLabels)
{
    if (SIZE_TRAINING_DATA == 0 || hostLabels == nullptr)
    {
        throw std::invalid_argument("Evaluation loss requires labels and a non-empty batch");
    }
    if (SIZE_TRAINING_DATA > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("Evaluation sample count exceeds CUDA launcher limits");
    }

    std::vector<float> labelCopy(
        hostLabels,
        hostLabels + SIZE_TRAINING_DATA);
    GVector deviceLabels(SIZE_TRAINING_DATA);
    deviceLabels.upload(labelCopy.data());
    return CudaLaunchers::sumCrossEntropyLoss(
        y_hat.getMatrix(),
        deviceLabels.getVector(),
        static_cast<int>(SIZE_TRAINING_DATA));
}