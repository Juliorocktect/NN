#include "VNN.h"

double NN::sigmoid(double x){
    if (x < -100.0) x = -100.0;
    if (x >  100.0) x =  100.0;
    return 1.0 / (1.0 + std::exp(-x));
}

Eigen::VectorXd NN::softmax(const Eigen::VectorXd& z) {
    Eigen::VectorXd exp_z = (z.array() - z.maxCoeff()).exp(); // für numerische Stabilität
    return exp_z / exp_z.sum();
}

Eigen::Matrix<double,1,10> forwardPropagation(std::vector<std::vector<uint8_t>>& images)
{
    
}
NN::NN()
{
    w1 = (Eigen::MatrixXd::Random(480, 784) + Eigen::MatrixXd::Constant(480, 784, 1.0)) / 2.0;
    w2 = (Eigen::MatrixXd::Random(200, 480) + Eigen::MatrixXd::Constant(200, 480, 1.0)) / 2.0;
    w3 = (Eigen::MatrixXd::Random(180,200) + Eigen::MatrixXd::Constant(180,200, 1.0)) / 2.0;
    w4 = (Eigen::MatrixXd::Random(10,180) + Eigen::MatrixXd::Constant(10,180, 1.0)) / 2.0;
    b1 = (Eigen::MatrixXd::Zero(480, 1));//Überflüssige Klammer?
    b2 = (Eigen::MatrixXd::Zero(200, 1));
    b3 = (Eigen::MatrixXd::Zero(180, 1)); 
    b4 = (Eigen::MatrixXd::Zero(10, 1)); 
}
// returns y_hat for avery iamg fed from the vector
Eigen::MatrixXd NN::forwardPropagation(std::vector<uint8_t> &image)//Nur ein Bild?
{
    //layer 1 calculations
    Eigen::MatrixXd A0(784,1);//Input Layer
    for (size_t i = 0; i < image.size(); ++i) {
        A0(i,0) = static_cast<double>(image[i]);
    }
    //A0  = A0.transpose().eval();//1x784
    Eigen::VectorXd Z1(480);//Ergebnis Matrix 480x1
    Z1 = w1 * A0 + b1;
    for(int i = 0; i < Z1.rows(); i++) {
        Z1(i) = sigmoid(Z1(i));
    }
   //layer 2 calc
   //layer 2: 200 Nodes
    Eigen:: MatrixXd Z2(200,1);// Ergebnis Layer 2 200x1
    Z2 = w2 * Z1 +b2;
    //Apply Sigmoid function
    for (int i = 0;i < Z2.rows();i++)
    {
        Z2(i) = sigmoid(Z2(i));
    }
    //Layer 3 Calulations
    Eigen::MatrixXd Z3(180,1); //Ergebnis Layer 3 180x1
    Z3 = w3 * Z2 + b3;
    //Apply Sigmoid
    for (int i = 0;i < Z3.rows();i++)
    {
        Z3(i) = sigmoid(Z3(i));
    }
    // Output Layer Calculus
    Eigen::MatrixXd y_hat(10,1);
    y_hat = w4 * Z3 + b4;
    y_hat = softmax(y_hat);
    return y_hat;//Nur für ein Bild die Berechnung
}
double NN::costF(double y,Eigen::VectorXd& y_hat)
{
    return (-1.0 * (y-(y_hat.array().log()) + (1-y)*(1.0 - y_hat.array()).log()));
}
