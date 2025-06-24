#include "VNN.h"

void NN::backpropagateOutputLayer(std::vector<uint8_t>& labels)//TODO: die Variablen Global speichern
{
    printGreen("Starte BackProp");
    double e = sumCrossEntropyLoss(labels);//Mean Squared Error Aus (y_hat - y)^2 wird y_hat - y
    //Ableitung vom Fehler nach y_hat
    dE_dYHAT = y_hat - y.transpose().eval();
    dW4 = (dE_dYHAT * A3.transpose().eval()) / SIZE_TRAINING_DATA;//10x180
    //Mittelwert über alle Trainingsdaten, weil nur ein b pro node und nicht pro Beispiel x Nodes
    db4 = dE_dYHAT.rowwise().mean();
    std::cout << "\033[1;32mOutput Layer Backpropagation Passed\033[0m\n";
}
void NN::backpropagateThirdLayer()
{
    printGreen("Starte Ableitung Layer 3");
    Eigen::MatrixXd dA3 = w4.transpose().eval() * dE_dYHAT;
    dYHAT_dZ3 = dA3.array() * Z3.unaryExpr([this](double x){return sigmoidDeriviative(x);}).eval().array();//Elementweise Multiplikation?
    dW3 = (dYHAT_dZ3 * Z2.transpose().eval()) / SIZE_TRAINING_DATA; // Muss ich Z2 durch A2 ersetzen, weil Z2 aktiviert wurde? //TODO: ja muss ich noch machen
    db3 = dYHAT_dZ3.rowwise().mean(); //bias durchschnitt Gradient
    printGreen("Layer 3 derived");
}
void NN::backpropagateSecondLayer()
{
    printGreen("Starte Ableitung Layer 2");
    Eigen::MatrixXd dA2 = w3.transpose().eval() * dYHAT_dZ3;
    dZ2 = dA2.array() * Z2.unaryExpr([this](double x){return sigmoidDeriviative(x);}).eval().array();
    dW2 = (dZ2 * Z1.transpose().eval()) / SIZE_TRAINING_DATA;
    db2 = dZ2.rowwise().mean();
    printGreen("Layer 2 derived");
}
void NN::backpropagateFirstLayer()
{
    printGreen("Starte Ableitung Layer 1");
    Eigen::MatrixXd dA1 = w2.transpose().eval() * dZ2;
    Eigen::MatrixXd dZ1 = dA1.array() * Z1.unaryExpr([this](double x){return sigmoidDeriviative(x);}).eval().array();
    dW1 = (dZ1 * inputData.transpose()) / SIZE_TRAINING_DATA;
    db1 = dZ1.rowwise().mean();
    printGreen("Layer 1 derived");
}