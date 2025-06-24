#ifndef VNN 
#define VNN
#pragma once
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <inttypes.h>
const int SIZE_INPUT_LAYER = 784;
const int SIZE_FIRST_LAYER = 480;
const int SIZE_SECOND_LAYER = 200;
const int SIZE_THIRD_LAYER = 180;
const int SIZE_TRAINING_DATA = 200;
const double LEARNING_RATE = 0.5;

double sigmoid(double x);
class NN
{
private:
    void printGreen(const char* text);
    void printRed(const char* text);
    Eigen::MatrixXd softmax(const Eigen::MatrixXd& m);
    double sigmoid(double x);
    double crossEntropyLoss(const Eigen::VectorXd& y_hat, uint8_t correct_label);// Andere Loss Function
    double sigmoidDeriviative(double x);
    void initilizeYMatrix(const std::vector<uint8_t>& labels);
    double costF(double y,const Eigen::VectorXd& y_hat);//Berechnet den Verlust vom Ergebnis zum eigentlichen Ergebnis
    Eigen::MatrixXd inputData;// 784x SIZE_TRAINING_DATA
    Eigen::MatrixXd w1;//480x784
    Eigen::MatrixXd w2;//200x480
    Eigen::MatrixXd w3;//180x200
    Eigen::MatrixXd w4;//10x180
    Eigen::MatrixXd Z1;//480xSIZE_TRAINING_DATA
    Eigen::MatrixXd Z2;//200xSIZE_TRAINING_DATA
    Eigen::MatrixXd Z3;//180xSIZE_TRAINING_DATA
    Eigen::MatrixXd Z4;//10xSIZE_TRAINING_DATA
    Eigen::MatrixXd A1;//480xSIZE_TRAINING_DATA
    Eigen::MatrixXd A2;//200xSIZE_TRAINING_DATA
    Eigen::MatrixXd A3;//180xSIZE_TRAINING_DATA
    Eigen::MatrixXd y_hat;//10xSIZE_TRRAINING_DATA
    Eigen::MatrixXd y;// 10xSIZE_TRAINING_DATA
    Eigen::MatrixXd b1; //bias 480x1
    Eigen::MatrixXd b2;//bias 200x1
    Eigen::MatrixXd b3;//bias 180x1 3. hidden layer
    Eigen::MatrixXd b4;//bias output layer 10x1
    //Deriviatives
    Eigen::MatrixXd dE_dYHAT;// Deriviative 10xSIZE_TRAINING_DATA Output Layer
    Eigen::MatrixXd dYHAT_dZ3;//Deriviative 180xSIZE_TRAINING_DATA Layer 3
    Eigen::MatrixXd dZ2;//Deriviative 200xSIZE_TRAINING_DATA Layer 2
    Eigen::MatrixXd db4;// Deriviative of bias mit einem Mittelwert 10x1
    Eigen::MatrixXd db3;// Deriviative of bias mit einem Mittelwert 180x1
    Eigen::MatrixXd db2;// Deriviative of bias mit einem Mittelwert 200x1
    Eigen::MatrixXd db1;// Deriviative of bias mit einem Mittelwert 480x1
    Eigen::MatrixXd dW1;//Deriviative of weights Layer 1 480x784
    Eigen::MatrixXd dW2;// Derivation of weights Layer 2 200x480
    Eigen::MatrixXd dW3;//Derivation of weights Layer 3 180x200
    Eigen::MatrixXd dW4;//Derivation of weights Output Layer 4 10x180
public:
    NN(std::vector<uint8_t>& labels);
    ~NN();
    void forwardPropagation();//berechnet y_hat mit der Batchgröße SIZE_TRAINING_DATA
    double sumCrossEntropyLoss(std::vector<uint8_t>& labels);// Returns summed losses
    void backpropagateOutputLayer(std::vector<uint8_t>& labels);// calculates the partial deriviatives for Output Layer 
    void backpropagateThirdLayer();//calculates the deriviative of Layer 3
    void backpropagateSecondLayer();//calculates the deriviatives of Layer 2
    void backpropagateFirstLayer();
    void setInputData(Eigen::MatrixXd& images);
};
#endif
