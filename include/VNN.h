#ifndef VNN 
#define VNN
#pragma once
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

double sigmoid(double x);
Eigen::Matrix<double,1,10> forwardPropagation(std::vector<std::vector<uint8_t>>& images);//Ka wie groß der Param sein soll

class NN
{
private:
    double sigmoid(double x);
    Eigen::Matrix<double,480,784> w1;
    Eigen::Vector<double,784> b1;
    Eigen::Vector<double,480> b2;
    Eigen::Vector<double,200> b3;
    Eigen::Vector<double,10> b4;
public:
    NN();
    ~NN();
    Eigen::Matrix<double,1,10> forwardPropagation(std::vector<uint8_t> &images);

};
#endif
