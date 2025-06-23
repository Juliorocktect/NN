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
    Eigen::MatrixXd softmax(const Eigen::MatrixXd& m);
    double sigmoid(double x);
    double crossEntropyLoss(const Eigen::VectorXd& y_hat, uint8_t correct_label);// Andere Loss Function
    double sigmoidDeriviative(double x);
    void initilizeYMatrix(const std::vector<uint8_t>& labels);
    double costF(double y,const Eigen::VectorXd& y_hat);//Berechnet den Verlust vom Ergebnis zum eigentlichen Ergebnis
    Eigen::MatrixXd w1;//480x784
    Eigen::MatrixXd w2;//200x480
    Eigen::MatrixXd w3;//180x200
    Eigen::MatrixXd w4;//10x180
    Eigen::MatrixXd Z1;//480xSIZE_TRAINING_DATA
    Eigen::MatrixXd Z2;//200xSIZE_TRAINING_DATA
    Eigen::MatrixXd Z3;//180xSIZE_TRAINING_DATA
    Eigen::MatrixXd y_hat;//10xSIZE_TRRAINING_DATA
    Eigen::MatrixXd y;// 10xSIZE_TRAINING_DATA
    Eigen::MatrixXd b1; //bias 480x1
    Eigen::MatrixXd b2;//bias 200x1
    Eigen::MatrixXd b3;//bias 180x1 3. hidden layer
    Eigen::MatrixXd b4;//bias output layer 10x1
    
public:
    NN(std::vector<uint8_t>& labels);
    ~NN();
    Eigen::MatrixXd forwardPropagation(Eigen::MatrixXd& images);//implementatnion mit eine Input-matrix der Trainingsdaten
    double sumCrossEntropyLoss(std::vector<uint8_t>& labels);// Returns summed losses
    void backpropagateOutputLayer(std::vector<uint8_t>& labels);// Returns the partial deriviatives for Layer 3
    // Gives Y the Value of the Vector in form of Matrices
};
#endif
