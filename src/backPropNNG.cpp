#include "VNN.h"

void NNG::backpropagateOutputLayer()
{
    printGreen("Starte BackProp");
    std::cout << y_hat.matrix[5*10+19000];
    //double e = sumCrossEntropyLoss();
    //std::cout << e << std::endl;
}