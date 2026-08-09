#ifndef VNN
#define VNN
#pragma once
#include <iostream>
#include <cmath>
#include <inttypes.h>
#include <Maths.h>
#define SIZE_INPUT_LAYER 784
#define SIZE_FIRST_LAYER 480
#define SIZE_SECOND_LAYER 200
#define SIZE_THIRD_LAYER 180
#define SIZE_TRAINING_DATA 40000
#define LEARNING_RATE 0.01

double sigmoid(double x);
class NNG
{
private:
    void printGreen(const char *text);
    void printRed(const char *text);
    double costF();

    GPUMatrix inputData; // 784xSIZE_TRAINING_DATA
    // Weights
    GPUMatrix w1; // 480x784
    GPUMatrix w2; // 200x480
    GPUMatrix w3; // 180x200;
    GPUMatrix w4; // 10x180;
    // Neurons
    GPUMatrix Z1;    // 480xSIZE_TRAINING_DATA
    GPUMatrix Z2;    // 200xSIZE_TRAINING_DATA
    GPUMatrix Z3;    // 180xSIZE_TRAINING_DATA
    GPUMatrix Z4;    // 10xSIZE_TRAINING_DATA
    GPUMatrix A1;    // 480xSIZE_TRAINING_DATA
    GPUMatrix A2;    // 200xSIZE_TRAINING_DATA
    GPUMatrix A3;    // 180xSIZE_TRAINING_DATA
    GPUMatrix y_hat; // 10xSIZE_TRAINING_DATA

    GPUMatrix y; // 10xSIZE_TRAINING_DATA Actual Result
    // Biases
    GPUMatrix b1; // bias 480x1
    GPUMatrix b2; // bias 200x1
    GPUMatrix b3; // bias 180x1 3. hidden layer
    GPUMatrix b4; // bias output layer 10x1
    // Derivatives
    GPUMatrix dE_dYHAT;  // Derivative 10xSIZE_TRAINING_DATA Output Layer
    GPUMatrix dYHAT_dZ3; // Derivative 180xSIZE_TRAINING_DATA Layer 3
    GPUMatrix dZ2;       // Derivative 200xSIZE_TRAINING_DATA Layer 2
    GPUMatrix db4;       // Derivative of bias mit einem Mittelwert 10x1
    GPUMatrix db3;       // Derivative of bias mit einem Mittelwert 180x1
    GPUMatrix db2;       // Derivative of bias mit einem Mittelwert 200x1
    GPUMatrix db1;       // Derivative of bias mit einem Mittelwert 480x1
    GPUMatrix dW1;       // Derivative of weights Layer 1 480x784
    GPUMatrix dW2;       // Derivation of weights Layer 2 200x480
    GPUMatrix dW3;       // Derivation of weights Layer 3 180x200
    GPUMatrix dW4;       // Derivation of weights Output Layer 4 10x180
    double *labels;      // 60.000 labels to the 60.000 images
public:
    NNG();
    ~NNG();
    void setInputData(double *mat);
    void setLabels(double *labels);
    void forwardProp();
    void backpropagateOutputLayer();
    void backpropagateThirdLayer();
    void backpropagateSecondLayer();
    void backpropagateFirstLayer();
    void updateWeightsAndBiases();
    void initilizeYMatrix(double *pLabels);
    double sumCrossEntropyLoss();
    int calcAccuracy(double *inputData);
    int argmax(const double *vec, int size);
};
#endif
