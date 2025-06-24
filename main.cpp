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
    }
    return 0;
}
