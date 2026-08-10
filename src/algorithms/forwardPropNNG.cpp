#include "VNN.h"

void NNG::forwardProp()
{
    // Layer 1 Calc
    // printGreen("Start Calc");

    Z1 = (w1 * inputData);
    Z1 = (Z1 + b1); //+ finished implementing?
    A1 = Z1.sigmoid();
    // printGreen("Layer 1 Passed");

    // Layer 2 Calc
    Z2 = (w2 * Z1);
    Z2 = (Z2 + b2);
    A2 = Z2.sigmoid();
    // printGreen("Layer 2 Passed");

    // Layer 3 Calc
    Z3 = (w3 * Z2);
    Z3 = (Z3 + b3);
    A3 = Z3.sigmoid();
    // printGreen("Layer 3 Passed");
    // Output Layer Calc
    y_hat = (w4 * Z3);
    y_hat = (y_hat + b4);
    y_hat.softmax();
    // printGreen("Output Layer Passed");
}