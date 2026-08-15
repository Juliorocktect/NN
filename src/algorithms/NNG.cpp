#include "VNN.h"

NNG::NNG()
{
    y = GMatrix(10, SIZE_TRAINING_DATA);
    w1 = GMatrix(480, 784);
    w1.initRandom();
    w2 = GMatrix(200, 480);
    w2.initRandom();
    w3 = GMatrix(180, 200);
    w3.initRandom();
    w4 = GMatrix(10, 180);
    w4.initRandom();
    A1 = GMatrix(SIZE_FIRST_LAYER, SIZE_TRAINING_DATA);
    A2 = GMatrix(SIZE_SECOND_LAYER, SIZE_TRAINING_DATA);
    A3 = GMatrix(SIZE_THIRD_LAYER, SIZE_TRAINING_DATA);
    Z1 = GMatrix(SIZE_FIRST_LAYER, SIZE_TRAINING_DATA); // Inkonsitnente Daten
    Z2 = GMatrix(SIZE_SECOND_LAYER, SIZE_TRAINING_DATA);
    Z3 = GMatrix(SIZE_THIRD_LAYER, SIZE_TRAINING_DATA);
    y_hat = GMatrix(10, SIZE_TRAINING_DATA);
    b1 = GVector(480);
    b2 = GVector(200);
    b3 = GVector(180);
    b4 = GVector(10);
    dE_dYHAT = GMatrix(10, SIZE_TRAINING_DATA);
    db4 = GVector(10);
    db3 = GVector(180);
    db2 = GVector(200);
    db1 = GVector(480);
    dYHAT_dZ3 = GMatrix(180, SIZE_TRAINING_DATA);
    dZ2 = GMatrix(200, SIZE_TRAINING_DATA);
    dW4 = GMatrix(10, 180);
    dW3 = GMatrix(180, 200);
    dW2 = GMatrix(200, 480);
    dW1 = GMatrix(480, 784);
    inputData = GMatrix(784, SIZE_TRAINING_DATA);
}
NNG::~NNG()
{
    free(labels);
}
void NNG::setInputData(float *mat)
{
    inputData.uploadMatrixToGPU(mat);
}
void NNG::printGreen(const char *text)
{
    std::cout << "\033[1;32m" << text << "\033[0m\n";
}
void NNG::printRed(const char *text)
{
    std::cout << "\033[1;33m" << text << "\033[0m\n";
}
void NNG::initilizeYMatrix(float *pLabels)
{
    this->labels = pLabels;
    // y muss mit 0 initialisiert sein!
    CudaLaunchers::hotEncodeYMatrix(pLabels, y.getMatrix(), SIZE_TRAINING_DATA);
}
void NNG::setLabels(float *labels)
{
    this->labels = labels;
}

double NNG::sumCrossEntropyLoss()
{
    // return executeCrossEntropyLoss(y_hat.matrix, labels, SIZE_TRAINING_DATA);
}
int NNG::argmax(const double *vec, int size)
{
    int maxIdx = 0;
    double maxVal = vec[0];
    for (int i = 1; i < size; ++i)
    {
        if (vec[i] > maxVal)
        {
            maxVal = vec[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}
/* int NNG::calcAccuracy(double *inputData)
{
    const int sizeUnusedImages = 60000 - SIZE_TRAINING_DATA;
    GMatrix unusedImaged(784, sizeUnusedImages);
    double m[sizeUnusedImages];
    std::copy(inputData + SIZE_TRAINING_DATA, inputData + 60000, m); // copy images from files into matrix
    unusedImaged.matrix = m;
    std::cout << "Executed copy of Matrix\n";
    // Feed Through net
    Z1 = (w1 * unusedImaged);
    Z1.addVectorColwise(b1); //+ finished implementing?
    A1 = Z1.sigmoid();
    // printGreen("Layer 1 Passed");

    // Layer 2 Calc
    Z2 = (w2 * Z1);
    Z2 = Z2 + b2;
    A2 = Z2.sigmoid();
    // printGreen("Layer 2 Passed");

    // Layer 3 Calc
    Z3 = (w3 * Z2);
    Z3.addVectorColwise(Z3);
    A3 = Z3.sigmoid();
    // printGreen("Layer 3 Passed");
    // Output Layer Calc
    y_hat = (w4 * Z3);
    y_hat.addVectorColwise(b4);
    y_hat.softmax();
    std::cout << "Feed through network\n";
    // execute Argmax Function
    double *matRes = new double[sizeUnusedImages]; // Predections of NN per picture
    matRes = executeArgmaxKernel(y_hat.matrix, y_hat.rows, y_hat.cols);
    std::cout << "Executed ArgmaxKernel\n";
    std::cout << matRes[12] << std::endl;
    std::cout << labels[8756] << std::endl;
    int correct = 0;
    for (int i = 0; i < sizeUnusedImages; i++) // Muss GPUOptimiert Werden//TODO:Probleme beim Lesen von labels falscher speicher zugriff
    {
        int predicted = matRes[i];
        int actual = labels[SIZE_TRAINING_DATA + i];
        if (predicted == actual)
            correct++;
    }
    delete[] matRes;
    return correct;
} */