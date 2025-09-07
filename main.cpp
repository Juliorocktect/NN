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
    GPUMatrix m2(2,3);
    m1.setMat(mat1);
    m2.setMat(mat2);
    GPUMatrix result = m1*m2;
    std::cout << std::endl << mat1 * mat2;
    std::cout << "\n";
    result.printMat();

    Eigen::MatrixXd mat3(2,2);
    Eigen::MatrixXd mat4(2,2);
    mat3 << 4,7,8,3;
    mat4 << 4,7,8,3;
    GPUMatrix m3(2,2);
    GPUMatrix m4(2,2);
    m3.setMat(mat3);
    m4.setMat(mat4);
    GPUMatrix add = m3+m4;
    add.printMat();
    return 0;
}
