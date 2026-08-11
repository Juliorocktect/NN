#include "VNN.h"

void NNG::backpropagateOutputLayer()
{
    // printGreen("Starte BackProp");
    double e = sumCrossEntropyLoss();
    std::cout << e << std::endl;
    dE_dYHAT = y_hat - y;
    GMatrix A3_t = A3.transpose();
    dW4 = dE_dYHAT * A3_t;
    // printGreen("Output Layer Backpropagated");
}

void NNG::backpropagateThirdLayer()
{
    // printGreen("Starte Ableitung Layer 3");
    GMatrix w4_t = w4.transpose(); // 180x10
    GMatrix dA3 = w4_t * dE_dYHAT; // 10x10
    dYHAT_dZ3 = Z3.sigmoidDeriviative();
    dYHAT_dZ3 = dA3.hadamardMultiplication(dYHAT_dZ3);
    GMatrix Z2_t = Z2.transpose();
    dW3 = (dYHAT_dZ3 * Z2_t) / SIZE_TRAINING_DATA;
    // db3 = dYHAT_dZ3.calcMeanFromMatrixRowise();
    //  printGreen("Layer 3 derived");
}

void NNG::backpropagateSecondLayer()
{
    // printGreen("Starte Ableitung Layer 2");
    GMatrix w3_t = w3.transpose();
    GMatrix dA2 = w3_t * dYHAT_dZ3;
    dZ2 = Z2.sigmoidDeriviative();
    dZ2 = dA2.hadamardMultiplication(dZ2);
    GMatrix Z1_t = Z1.transpose();
    dW2 = (dZ2 * Z1_t) / SIZE_TRAINING_DATA;
    // db2 = dZ2.calcMeanFromMatrixRowise();
    //  printGreen("Layer 2 derived");
}
void NNG::backpropagateFirstLayer()
{
    // printGreen("Starte Ableitung Layer 1");
    GMatrix w2_t = w2.transpose();
    GMatrix dA1 = w2_t * dZ2;
    GMatrix dZ1 = Z1.sigmoidDeriviative();
    dZ1 = dA1.hadamardMultiplication(dZ1);
    GMatrix input_t = inputData.transpose();
    dW1 = (dZ1 * input_t) / SIZE_TRAINING_DATA;
    // db1 = dZ1.calcMeanFromMatrixRowise();
    //  printGreen("Layer 1 derived");
}
void NNG::updateWeightsAndBiases()
{
    // printGreen("Updating Weights");
    // Update weights
    w4 = w4 - dW4.multiplicationSingleV(LEARNING_RATE);
    w3 = w3 - dW3.multiplicationSingleV(LEARNING_RATE);
    w2 = w2 - dW2.multiplicationSingleV(LEARNING_RATE);
    w1 = w1 - dW1.multiplicationSingleV(LEARNING_RATE);
    // printGreen("Updating Biases");
    // update biases
    b4 = b4.vectorSub(db4.multiplicationSingleV(LEARNING_RATE));
    b3 = b3.vectorSub(db3.multiplicationSingleV(LEARNING_RATE));
    b2 = b2.vectorSub(db2.multiplicationSingleV(LEARNING_RATE));
    b1 = b1.vectorSub(db1.multiplicationSingleV(LEARNING_RATE));
}