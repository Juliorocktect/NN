#ifndef VNN
#define VNN
#pragma once
#include <iostream>
#include <cmath>
#include <inttypes.h>
#include <Maths.h>
#include "GMatrix.hpp"
#include "GVector.hpp"
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

    GMatrix inputData; // 784xSIZE_TRAINING_DATA
    // Weights
    GMatrix w1; // 480x784
    GMatrix w2; // 200x480
    GMatrix w3; // 180x200;
    GMatrix w4; // 10x180;
    // Neurons
    GMatrix Z1;    // 480xSIZE_TRAINING_DATA
    GMatrix Z2;    // 200xSIZE_TRAINING_DATA
    GMatrix Z3;    // 180xSIZE_TRAINING_DATA
    GMatrix Z4;    // 10xSIZE_TRAINING_DATA
    GMatrix A1;    // 480xSIZE_TRAINING_DATA
    GMatrix A2;    // 200xSIZE_TRAINING_DATA
    GMatrix A3;    // 180xSIZE_TRAINING_DATA
    GMatrix y_hat; // 10xSIZE_TRAINING_DATA

    GMatrix y; // 10xSIZE_TRAINING_DATA Actual Result
    // Biases
    GVector b1; // bias 480x1
    GVector b2; // bias 200x1
    GVector b3; // bias 180x1 3. hidden layer
    GVector b4; // bias output layer 10x1
    // Derivatives
    GMatrix dE_dYHAT;  // Derivative 10xSIZE_TRAINING_DATA Output Layer
    GMatrix dYHAT_dZ3; // Derivative 180xSIZE_TRAINING_DATA Layer 3
    GMatrix dZ2;       // Derivative 200xSIZE_TRAINING_DATA Layer 2
    GVector db4;       // Derivative of bias mit einem Mittelwert 10x1
    GVector db3;       // Derivative of bias mit einem Mittelwert 180x1
    GVector db2;       // Derivative of bias mit einem Mittelwert 200x1
    GVector db1;       // Derivative of bias mit einem Mittelwert 480x1
    GMatrix dW1;       // Derivative of weights Layer 1 480x784
    GMatrix dW2;       // Derivation of weights Layer 2 200x480
    GMatrix dW3;       // Derivation of weights Layer 3 180x200
    GMatrix dW4;       // Derivation of weights Output Layer 4 10x180
    float *labels;    // 60.000 labels to the 60.000 images
public:
    NNG();
    ~NNG();
    void setInputData(float *mat);
    void setLabels(double *labels);
    void forwardProp();
    void backpropagateOutputLayer();
    void backpropagateThirdLayer();
    void backpropagateSecondLayer();
    void backpropagateFirstLayer();
    void updateWeightsAndBiases();
    void initilizeYMatrix(float *pLabels);
    double sumCrossEntropyLoss();
    // int calcAccuracy(float *inputData);
    int argmax(const double *vec, int size);
};
#endif
