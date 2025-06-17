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


double sigmoid(double x);
class NN
{
private:
    Eigen::VectorXd softmax(const Eigen::VectorXd& z);
    double sigmoid(double x);
    Eigen::MatrixXd w1;//480x784
    Eigen::MatrixXd w2;//200x480
    Eigen::MatrixXd w3;//180x200
    Eigen::MatrixXd w4;//10x180
    Eigen::MatrixXd output;//output 10x1
    Eigen::MatrixXd b1; //bias 480x1
    Eigen::MatrixXd b2;//bias 200x1
    Eigen::MatrixXd b3;//bias 180x1 3. hidden layer
    Eigen::MatrixXd b4;//bias output layer 10x1
public:
    NN();
    ~NN();
    Eigen::MatrixXd forwardPropagation(std::vector<uint8_t> &image);
    double lossF();

};
#endif
