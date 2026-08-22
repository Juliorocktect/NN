#ifndef IMAGE_LOAD_H
#define IMAGE_LOAD_H
#include <fstream>
#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <vector>
#include <SDL3/SDL.h>
#include <limits>
#define IMAGE_PATH "/home/julio/Dokumente/NN/resources/train-images.idx3-ubyte"
#define LABEL_PATH "/home/julio/Dokumente/NN/resources/train-labels.idx1-ubyte"
#define IMAGE_TEST_PATH "/home/julio/Dokumente/NN/resources/t10k-images.idx3-ubyte"
#define LABEL_TEST_PATH "/home/julio/Dokumente/NN/resources/t10k-labels.idx1-ubyte"
namespace ImagePreProcessor
{
    int readInt(std::ifstream &ifs);
    std::vector<std::vector<uint8_t>> readImages();
    std::vector<uint8_t> readLabels();
    void showImage(const std::vector<uint8_t> &pixels, int width, int height);
    void showImage(const float *pixels, int width, int height);
    double *loadImages();
    std::vector<double> readLabelsAsDouble();
    /**
     * @brief loads up to the specified number of images and normalizes them to [0,1].
     *
     * @param imageCount Maximum number of images to load. If 0, all available images are loaded.
     * @return float* with normalized pixel data.
     */
    float *loadImageAsFLoat(size_t imageCount);

    /**
     * @brief loads labels to the image idx = idx
     * not hot encoded!
     *
     * @return std::vector<float>
     */
    std::vector<float> readLabelsAsFloat();

    /**
     * @brief return matrix 784x10.000 of test images
     *
     * @return float*
     */
    float *loadImageTestData(size_t imageCount);

    /**
     * @brief loads the Labels in a vector 1x10.000
     *
     * @return float*
     */
    float *readTestLabels();
}
#endif