#include "VNN.h"

NNG::NNG()
{
    w1 = GPUMatrix(480,784);
    w1.init();
    w2 = GPUMatrix(200,480);
    w2.init();
    w3 = GPUMatrix(180,200);
    w3.init();
    w4 = GPUMatrix(10,180);
    w4.init();
    A1 = GPUMatrix(SIZE_FIRST_LAYER,SIZE_TRAINING_DATA);
    A2 = GPUMatrix(SIZE_SECOND_LAYER,SIZE_TRAINING_DATA);
    A3 = GPUMatrix(SIZE_THIRD_LAYER,SIZE_TRAINING_DATA);
    Z1  =GPUMatrix(SIZE_FIRST_LAYER,SIZE_TRAINING_DATA);//Inkonsitnente Daten
    Z2 = GPUMatrix(SIZE_SECOND_LAYER, SIZE_TRAINING_DATA);
    Z3 = GPUMatrix(SIZE_THIRD_LAYER, SIZE_TRAINING_DATA);
    y_hat = GPUMatrix(10,SIZE_TRAINING_DATA);
    
    y_hat.initZero();
    b1 = GPUMatrix(480,1);
    b1.initZero();
    b2 = GPUMatrix(200,1);
    b2.initZero();
    b3 = GPUMatrix(180,1);
    b3.initZero();
    b4 = GPUMatrix(10,1);
    b4.initZero();
    dE_dYHAT = GPUMatrix(10,SIZE_TRAINING_DATA);
    dE_dYHAT.initZero();
    db4 = GPUMatrix(10, 1);
    db3 = GPUMatrix(180, 1);
    db2 = GPUMatrix(200, 1);
    db1 = GPUMatrix(480, 1);
    db1.initZero();
    db2.initZero();
    db3.initZero();
    db4.initZero();
    dYHAT_dZ3 = GPUMatrix(180,SIZE_TRAINING_DATA);
    dZ2 = GPUMatrix(200,SIZE_TRAINING_DATA);
    dW4 = GPUMatrix(10,180);
    dW3 = GPUMatrix(180,200);
    dW2 = GPUMatrix(200,480);
    dW1 = GPUMatrix(480,784);
    inputData = GPUMatrix(784,SIZE_TRAINING_DATA);
}
void NNG::setInputData(double* mat)
{
    inputData.matrix = mat;
}
void NNG::printGreen(const char* text)
{
    std::cout << "\033[1;32m"<< text << "\033[0m\n";
}
void NNG::printRed(const char* text)
{
    std::cout << "\033[1;33m"<< text << "\033[0m\n";
}
void NNG::initilizeYMatrix(double* pLabels)
{
    this->labels = labels; 
    y = GPUMatrix(10,SIZE_TRAINING_DATA);
    y.matrix = hotEncodeYMatrix(pLabels,SIZE_TRAINING_DATA);
}
void NNG::setLabels(double* labels)
{
    this->labels = labels;
}

double NNG::sumCrossEntropyLoss()
{
    double sum = 0.0;
    for (size_t i = 0;i < SIZE_TRAINING_DATA;i++)   
    {
        //sum += crossEntropyLoss(y_hat.col(i),labels[i]); // non quadratic
        int label = static_cast<int>(labels[i]);
        if (label < 0 || label >= 10) 
        {
            sum += -std::log(y_hat.matrix[i * 10 + label]);
        }
        //10xSIZE_TRAINIG_DATA
    }
    return sum/SIZE_TRAINING_DATA;
}