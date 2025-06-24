#include "VNN.h"

double NN::sigmoid(double x){
    if (x < -100.0) x = -100.0;
    if (x >  100.0) x =  100.0;
    return 1.0 / (1.0 + std::exp(-x));
}
double NN::sigmoidDeriviative(double x)
{
    double y = sigmoid(x);
    return y * (1 - y); 
}
void NN::printGreen(const char* text)
{
    std::cout << "\033[1;32m"<< text << "\033[0m\n";
}
void NN::printRed(const char* text)
{
    std::cout << "\033[1;33m"<< text << "\033[0m\n";
}
Eigen::MatrixXd NN::softmax(const Eigen::MatrixXd& m) 
{
    Eigen::MatrixXd result = m;
    for (int col = 0; col < m.cols(); ++col) {
        Eigen::VectorXd z = m.col(col);
        Eigen::VectorXd exp_z = (z.array() - z.maxCoeff()).exp();
        result.col(col) = exp_z / exp_z.sum();
    }
    return result;
}
Eigen::Matrix<double,1,10> forwardPropagation(std::vector<std::vector<uint8_t>>& images)
{
    
}
NN::NN(std::vector<uint8_t>& labels)
{
    w1 = Eigen::MatrixXd::Random(480, 784) * 0.01;
    w2 = Eigen::MatrixXd::Random(200, 480) * 0.01;
    w3 = Eigen::MatrixXd::Random(180, 200) * 0.01;
    w4 = Eigen::MatrixXd::Random(10, 180) * 0.01;
    A1 = Eigen::MatrixXd(SIZE_FIRST_LAYER,SIZE_TRAINING_DATA);
    A2 = Eigen::MatrixXd(SIZE_SECOND_LAYER,SIZE_TRAINING_DATA);
    A3 = Eigen::MatrixXd(SIZE_THIRD_LAYER,SIZE_TRAINING_DATA);
    Z1 = Eigen::MatrixXd(SIZE_FIRST_LAYER,SIZE_TRAINING_DATA);
    Z2 = Eigen::MatrixXd(SIZE_SECOND_LAYER,SIZE_TRAINING_DATA);
    Z3 = Eigen::MatrixXd(SIZE_THIRD_LAYER,SIZE_TRAINING_DATA);
    y_hat = Eigen::MatrixXd(10, SIZE_TRAINING_DATA);
    b1 = (Eigen::MatrixXd::Zero(480, 1));//Überflüssige Klammer?
    b2 = (Eigen::MatrixXd::Zero(200, 1));
    b3 = (Eigen::MatrixXd::Zero(180, 1)); 
    b4 = (Eigen::MatrixXd::Zero(10, 1)); 
    dE_dYHAT = Eigen::MatrixXd::Zero(10,SIZE_TRAINING_DATA);
    db4 = Eigen::MatrixXd::Zero(10,1);
    db3 = Eigen::MatrixXd::Zero(180,1);
    db2 = Eigen::MatrixXd::Zero(200,1);
    db1 = Eigen::MatrixXd::Zero(480,1);
    dYHAT_dZ3 = Eigen::MatrixXd(180,SIZE_TRAINING_DATA);
    dZ2 = Eigen::MatrixXd(200,SIZE_TRAINING_DATA);
    dW4 = Eigen::MatrixXd(10,180);
    dW3 = Eigen::MatrixXd(180,200);
    dW2 = Eigen::MatrixXd(200,480);
    dW1 = Eigen::MatrixXd(480,784);
    inputData = Eigen::MatrixXd(784,SIZE_TRAINING_DATA);

    initilizeYMatrix(labels);
}
void NN::initilizeYMatrix(const std::vector<uint8_t>& labels)
{
    y = Eigen::MatrixXd::Zero(SIZE_TRAINING_DATA, 10); // (Zeilen: Bilder, Spalten: Klassen)
    for (size_t i = 0; i < SIZE_TRAINING_DATA; i++)
    {
        if (labels[i] < 10) // Sicherheitscheck für MNIST
            y(i, labels[i]) = 1.0;
    }
}
void NN::setInputData(Eigen::MatrixXd& images)
{
    inputData = images;
}


double NN::costF(double y,const Eigen::VectorXd& y_hat)
{
    // Assuming y is the index of the correct class (for classification)
    // and y_hat is a probability vector (output of softmax)
    // Use cross-entropy loss for a single sample:
    // -log(y_hat[y])
    int label = static_cast<int>(y);
    if (label < 0 || label >= y_hat.size()) return 0.0;
    return -std::log(y_hat(label));
}
double NN::crossEntropyLoss(const Eigen::VectorXd& y_hat, uint8_t correct_label) 
{
    // Sicherheitsprüfung
    if (correct_label < 0 || correct_label >= y_hat.size()) return 0.0;

    // Kleine Konstante, um log(0) zu vermeiden
    const double epsilon = 1e-15;

    // Clamping der vorhergesagten Wahrscheinlichkeit
    double prob = std::clamp(y_hat(correct_label), epsilon, 1.0);

    return -std::log(prob);
}//Hier von den Gradienten Bilden
double NN::sumCrossEntropyLoss(std::vector<uint8_t>& labels)
{
    /* if (y_hat.cols() != labels.size())
    {
        std::cerr << "unenven Values\n";
    } */
    double sum = 0.0;
    for (size_t i = 0;i < y_hat.cols();i++)
    {
        //sum += crossEntropyLoss(y_hat.col(i),labels[i]); // non quadratic
        sum += costF(labels[i],y_hat.col(i));
    }
    return sum/SIZE_TRAINING_DATA;
}
