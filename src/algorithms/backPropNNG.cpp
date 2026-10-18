#include "VNN.h"

void NNG::backpropagateOutputLayer()
{
    float e = sumCrossEntropyLoss();
    std::cout << e << std::endl;
    dE_dYHAT = y_hat - y;
    GMatrix A3_t = A3.transpose();
    dW4 = (dE_dYHAT * A3_t) / SIZE_TRAINING_DATA;
    db4 = dE_dYHAT.calcMeanRowise();
}

void NNG::backpropagateThirdLayer()
{
    GMatrix w4_t = w4.transpose(); // 180x10
    GMatrix dA3 = w4_t * dE_dYHAT; // 10x10
    dYHAT_dZ3 = Z3.sigmoidDeriviative();
    dYHAT_dZ3 = dA3.hadamardMultiplication(dYHAT_dZ3);
    GMatrix A2_t = A2.transpose();
    dW3 = (dYHAT_dZ3 * A2_t) / SIZE_TRAINING_DATA;
    db3 = dYHAT_dZ3.calcMeanRowise();
}

void NNG::backpropagateSecondLayer()
{
    GMatrix w3_t = w3.transpose();
    GMatrix dA2 = w3_t * dYHAT_dZ3;
    dZ2 = Z2.sigmoidDeriviative();
    dZ2 = dA2.hadamardMultiplication(dZ2);
    GMatrix A1_t = A1.transpose();
    dW2 = (dZ2 * A1_t) / SIZE_TRAINING_DATA;
    db2 = dZ2.calcMeanRowise();
}
void NNG::backpropagateFirstLayer()
{
    GMatrix A1_t = A1.transpose();
    GMatrix w2_t = w2.transpose();
    GMatrix dA1 = w2_t * dZ2;
    GMatrix dZ1 = Z1.sigmoidDeriviative();
    dZ1 = dA1.hadamardMultiplication(dZ1);
    GMatrix input_t = inputData.transpose();
    dW1 = (dZ1 * input_t) / SIZE_TRAINING_DATA;
    db1 = dZ1.calcMeanRowise();
}
void NNG::updateWeightsAndBiases()
{
    GMatrix tmpW4 = (dW4 * LEARNING_RATE);
    w4 = w4 - tmpW4;
    GMatrix tmpW3 = (dW3 * LEARNING_RATE);
    w3 = w3 - tmpW3;
    GMatrix tmpW2 = (dW2 * LEARNING_RATE);
    w2 = w2 - tmpW2;
    GMatrix tmpW1 = (dW1 * LEARNING_RATE);
    w1 = w1 - tmpW1;
    // printGreen("Updating Biases");
    // update biases
    GVector tmpB4 = (db4 * LEARNING_RATE);
    b4 = b4 - tmpB4;
    GVector tmpB3 = (db3 * LEARNING_RATE);
    b3 = b3 - tmpB3;
    GVector tmpB2 = (db2 * LEARNING_RATE);
    b2 = b2 - tmpB2;
    GVector tmpB1 = (db1 * LEARNING_RATE);
    b1 = b1 - tmpB1;
}