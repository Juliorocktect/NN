#include <VNN.h>
#include <ImageLoading.h>
#include <vector>
#include <string>
#include <Maths.h>
#include <cuda_runtime.h>
#include "GMatrix.hpp"
#include <iostream>

int main(int argc, char const *argv[])
{
    NNG *n = new NNG();
    std::vector<float> lbl = ImagePreProcessor::readLabelsAsFloat();
    std::cout << lbl.at(9023) << std::endl;

    n->initilizeYMatrix(lbl.data());

    float *images = ImagePreProcessor::loadImageAsFLoat(40000);
    n->setInputData(images);
    for (int i = 0; i < 20000; i++)
    {
        n->forwardProp();
        n->backpropagateOutputLayer();
        n->backpropagateThirdLayer();
        n->backpropagateFirstLayer();
        n->updateWeightsAndBiases();
    }

    // TODO:calc accuaracy and loss
    return 0;
}
