#ifndef VNN 
#define VNN
#pragma once
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

double sigmoid(double x);
void compileMsg();

Eigen::Matrix<double,1,10> forwardPropagation();//Ka wie groß der Param sein soll

#endif
