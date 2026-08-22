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
    NNG *n = new NNG(40000);
    std::vector<float> lbl = ImagePreProcessor::readLabelsAsFloat();
    n->initilizeYMatrix(lbl.data());
    float *images = ImagePreProcessor::loadImageAsFLoat(40000);
    /*  std::vector<float> testImage(images + 784 * 1200, images + 784 * 1201);
     ImagePreProcessor::showImage(testImage.data(), 28, 28);
     std::cout << lbl.at(1200) << std::endl; */
    n->setInputData(images);
    n->run(20000);

    float *testImages = ImagePreProcessor::loadImageTestData(10000);
    float *testLabels = ImagePreProcessor::readTestLabels();
    // n->calcAccuracy(testImages, testLabels, 10000);
    // TODO:calc accuaracy and loss

    return 0;
}