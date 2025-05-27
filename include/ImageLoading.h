#ifndef IMAGE_LOAD_H
#define IMAGE_LOAD_H
#include <fstream> 
#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <vector>
#include <SDL3/SDL.h>
#include <Eigen/Dense>


namespace ImagePreProcessor
{
    int readInt(std::ifstream &ifs);
    std::vector<std::vector<uint8_t>> readImages();
    std::vector<uint8_t> readLabels();
    void showImage(const std::vector<uint8_t>& pixels, int width, int height); 
}
#endif