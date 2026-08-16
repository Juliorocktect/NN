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
    n->forwardProp();
    n->backpropagateOutputLayer();
    //    for (int i = 0; i < 15; i++)
    //{
    //    n->forwardProp();
    //     n->backpropagateOutputLayer();
    //     n->backpropagateThirdLayer();
    //     n->backpropagateSecondLayer();
    //     n->backpropagateFirstLayer();
    //     n->updateWeightsAndBiases();
    //   }
    //    int c = n->calcAccuracy(ImagePreProcessor::loadImages());
    //    std::cout << (double) c/20.000;
    return 0;
}
