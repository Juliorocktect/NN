#include <VNN.h>
#include <ImageLoading.h>
#include <Eigen/Dense>
#include <vector>
#include <string>

int main(int argc, char const *argv[])
{
    std::vector<uint8_t> labels = ImagePreProcessor::readLabels();
    std::vector<std::vector<uint8_t>> images = ImagePreProcessor::readImages();
    //std::cout << static_cast<int>(labels[24]) << std::endl;
    //ImagePreProcessor::showImage(images[24],28,28);
    NN* n = new NN(labels);
     Eigen::MatrixXd result(10,200);
    Eigen::MatrixXd inputMatrix(784,200);
    for(size_t j = 0; j < 200 ;j++)
    {
        for (size_t i = 0; i < images[0].size(); ++i)
        {
            inputMatrix(i,j) = static_cast<double>(images[j][i]);
        }
    }
    result = n->forwardPropagation(inputMatrix);//feed all training data, backpropagate, update, again
    std::cout << result.col(2) << std::endl;
    std::cout << "cost after one cycle\t" << n->sumCrossEntropyLoss(labels) << std::endl;
    n->backpropagateOutputLayer(labels);
    return 0;
}
