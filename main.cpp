#include <VNN.h>
#include <ImageLoading.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <Maths.h>
#include <cuda_runtime.h>


int main(int argc, char const *argv[])
{
    /* std::vector<uint8_t> labels = ImagePreProcessor::readLabels();
    std::vector<std::vector<uint8_t>> images = ImagePreProcessor::readImages();
    //std::cout << static_cast<int>(labels[24]) << std::endl;
    //ImagePreProcessor::showImage(images[24],28,28);
    NN* n = new NN(labels);
    Eigen::MatrixXd inputMatrix(784,2000);
    for(size_t j = 0; j < 2000 ;j++)
    {
        for (size_t i = 0; i < images[0].size(); ++i)
        {
            inputMatrix(i,j) = static_cast<double>(images[j][i]);
        }
    }
    n->setInputData(inputMatrix);
    for (int i = 0;i < 300;i++)
    {
        n->forwardPropagation();//feed all training data, backpropagate, update, again
        std::cout << "cost after one cycle\t" << n->sumCrossEntropyLoss(labels) << std::endl;
        n->backpropagateOutputLayer(labels);
        n->backpropagateThirdLayer();
        n->backpropagateSecondLayer();
        n-> backpropagateFirstLayer();
        n->updateWeightsAndBiases();
    } */
    GPUMatrix m1(3,2);
    Eigen::MatrixXd mat1(3,2);
    Eigen::MatrixXd mat2(2,3);
    mat1 << 7,5,98,32,2,8;
    mat2 << 8,90,3,2,1,9;
    double data_m1[] = {7,5,98,32,2,8};
    double data_m2[] = {8,90,3,2,1,9};
    GPUMatrix m2(2,3);
    m1.matrix = data_m1;
    m2.matrix = data_m2;
    GPUMatrix result = m1*m2;
    result.printMat();
    Eigen::MatrixXd  nr = mat1 * mat2;
    std::cout << "\n";
    result.transpose();
    result.printMat();
    std::cout << nr.transpose().eval() << std::endl;
    double matrix[9] = {2.5, 1.2, 3.7,
                        0.9, 4.1, 2.2,
                        3.3, 0.5, 1.8}; 
    double* r = execcuteSoftMaxKernel(matrix,3,3);
    GPUMatrix neu(3,3);
    std::cout << "\n";
    neu.matrix = r;
    neu.printMat();
    return 0;
}
