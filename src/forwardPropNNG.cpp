#include "VNN.h"

void NNG::forwardProp()
{
    //Layer 1 Calc
    std::cout << w1.rows << "\t" << w1.cols<<"\n";
    printGreen("Start Calc");
    Z1 = (w1 * inputData) + b1;//+ finished implementing?
    A1 = Z1.sigmoid();
    printGreen("Layer 1 Passed");

    //Layer 2 Calc
    Z2 = (w2 * Z1) + b2;
    A2 = Z2.sigmoid();
    printGreen("Layer 2 Passed");

    //Layer 3 Calc
    Z3 = (w3*Z2) + b3;
    A3 = Z3.sigmoid();
    printGreen("Layer 3 Passed");

    //Output Layer Calc
    std::cout << y_hat.rows << "\t" << y_hat.cols<<"\n";
    y_hat = (w4* Z3) + b4;
    //y_hat.softmax();
    std::cout << y_hat.rows << "\t" << y_hat.cols<<"\n";
    printGreen("Output Layer Passed");
}