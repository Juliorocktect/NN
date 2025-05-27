#include "VNN.h"

double sigmoid(double x){
    return exp(x)/(1 +exp(x));
}

Eigen::Matrix<double,1,10> forwardPropagation(std::vector<std::vector<uint8_t>>& images)
{
    
}

Eigen::Matrix<double, 1, 10> NN::forwardPropagation(std::vector<uint8_t> &images)
{
    //layer 1 calculations
    Eigen::VectorXd A0(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        A0(i) = static_cast<double>(images[i]);
    }
    Eigen::VectorXd Z1(480);
    Z1 = w1 * A0 + b1;
    for(int i = 0; i < Z1.rows(); i++) {
        Z1(i) = sigmoid(Z1(i));
    }
    //layer 2 calc
    
    

    return Eigen::Matrix<double, 1, 10>();
}
