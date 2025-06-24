#include "VNN.h"

void NN::forwardPropagation()
{
    //layer 1 calculations
    //A0  = A0.transpose().eval();//1x784
    printGreen("Start Calculating");
    //Ergebnis Matrix 480xSIZE_TRAINING_DATA
    Z1 = (w1 * inputData).colwise() + b1.col(0);
    std::cout << Z1.rows() << std::endl;
    for(int i = 0; i < A1.rows(); i++) {
        A1(i) = sigmoid(Z1(i));
    }
    printGreen("Layer 1 passed");
   //layer 2 calc
   //layer 2: 200 Nodes
    Z2 = (w2 * Z1).colwise() + b2.col(0);
    //Apply Sigmoid function
    for (int i = 0;i < A2.rows();i++)
    {
        A2(i) = sigmoid(Z2(i));
    }
    printGreen("Layer 2 passed");
    //Layer 3 Calulations
    //Ergebnis Layer 3 180xSIZE_TRAINING_DATA
    Z3 = (w3 * Z2).colwise() + b3.col(0);
    //Apply Sigmoid
    for (int i = 0;i < A3.rows();i++)
    {
        A3(i) = sigmoid(Z3(i));
    }
    printGreen("Layer 3 passed");
    // Output Layer Calculus
    y_hat = (w4 * Z3).colwise() + b4.col(0);
    y_hat = softmax(y_hat);
    printGreen("Output Layer passed");
    return;
}