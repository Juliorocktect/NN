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
    Z1 = Eigen::MatrixXd(SIZE_FIRST_LAYER,SIZE_TRAINING_DATA);
    Z2 = Eigen::MatrixXd(SIZE_SECOND_LAYER,SIZE_TRAINING_DATA);
    Z3 = Eigen::MatrixXd(SIZE_THIRD_LAYER,SIZE_TRAINING_DATA);
    y_hat = Eigen::MatrixXd(SIZE_TRAINING_DATA,10);
    y = Eigen::MatrixXd::Zero(SIZE_TRAINING_DATA,10);
    b1 = (Eigen::MatrixXd::Zero(480, 1));//Überflüssige Klammer?
    b2 = (Eigen::MatrixXd::Zero(200, 1));
    b3 = (Eigen::MatrixXd::Zero(180, 1)); 
    b4 = (Eigen::MatrixXd::Zero(10, 1)); 
    initilizeYMatrix(labels);
}
void NN::initilizeYMatrix(const std::vector<uint8_t>& labels)
{
    for(size_t i = 0;i < SIZE_TRAINING_DATA;i++)
    {
        y(i,labels[i]) = 1.0;
    }
    
}
// returns y_hat for avery iamg fed from the vector
Eigen::MatrixXd NN::forwardPropagation(Eigen::MatrixXd& images)//@Param 784x40.000
{
    //layer 1 calculations
    //A0  = A0.transpose().eval();//1x784
    std::cout << "Start Calculating\n";
    //Ergebnis Matrix 480xSIZE_TRAINING_DATA
    Z1 = (w1 * images).colwise() + b1.col(0);
    std::cout << Z1.rows() << std::endl;
    for(int i = 0; i < Z1.rows(); i++) {
        Z1(i) = sigmoid(Z1(i));
    }
    std::cout << "Layer 1 Passed\n";
   //layer 2 calc
   //layer 2: 200 Nodes
    Z2 = (w2 * Z1).colwise() + b2.col(0);
    //Apply Sigmoid function
    for (int i = 0;i < Z2.rows();i++)
    {
        Z2(i) = sigmoid(Z2(i));
    }
    std::cout << "Layer 2 Passed\n";
    //Layer 3 Calulations
    //Ergebnis Layer 3 180xSIZE_TRAINING_DATA
    Z3 = (w3 * Z2).colwise() + b3.col(0);
    //Apply Sigmoid
    for (int i = 0;i < Z3.rows();i++)
    {
        Z3(i) = sigmoid(Z3(i));
    }
    std::cout << "Layer 3 Passed\n";
    // Output Layer Calculus
    y_hat = (w4 * Z3).colwise() + b4.col(0);
    y_hat = softmax(y_hat);
    std::cout << "Output Layer Passed\n"; 
    return y_hat;//Ein Durchlauf
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
void NN::backpropagateOutputLayer(std::vector<uint8_t>& labels)
{
    std::cout << "Starte BackPropagation\n";
    double e = sumCrossEntropyLoss(labels);//Mean Squared Error Aus (y_hat - y)^2 wird y_hat - y
    //Ableitung vom Fehler nach y_hat
    Eigen::MatrixXd dE_dYHAT(10,200);//Matrix 10x200
    std::cout << "y_hat: " << y_hat.rows() << "x" << y_hat.cols() << std::endl;
    std::cout << "y: " << y.rows() << "x" << y.cols() << std::endl;
    dE_dYHAT = y_hat - y;
    std::cout << dE_dYHAT.col(0);
}