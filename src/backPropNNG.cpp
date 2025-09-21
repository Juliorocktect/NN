#include "VNN.h"

void NNG::backpropagateOutputLayer()
{
    printGreen("Starte BackProp");
    std::cout << y_hat.matrix[5*10+19000];
    //double e = sumCrossEntropyLoss();
    //std::cout << e << std::endl;
    dE_dYHAT = y_hat - y;
    GPUMatrix A3_t = A3.transpose();
    dW4 = dE_dYHAT * A3_t;
    printGreen("Output Layer Backpropagated");
}

void NNG::backpropagateThirdLayer()
{
    printGreen("Starte Ableitung Layer 3");
    GPUMatrix w4_t = w4.transpose();//180x10
    GPUMatrix dA3 =  w4_t * dE_dYHAT;//10x10
    Z3 = Z3.sigmoidDeriviative();
    dYHAT_dZ3 = dA3.hadamardMultiplication(Z3);
    //dW3 = dYHAT_dZ3 * Z2.transpose() / SIZE_TRAINING_DATA;
    printGreen("Layer 3 derived");
}