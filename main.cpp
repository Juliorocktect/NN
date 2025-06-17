#include <VNN.h>
#include <ImageLoading.h>
#include <Eigen/Dense>
#include <vector>
#include <string>

int main(int argc, char const *argv[])
{
    std::vector<uint8_t> labels = ImagePreProcessor::readLabels();
    std::vector<std::vector<uint8_t>> images = ImagePreProcessor::readImages();
    std::cout << static_cast<int>(labels[24]) << std::endl;
    //ImagePreProcessor::showImage(images[24],28,28);
    NN* n = new NN();
    std::cout << images[24].size() << std::endl;
     Eigen::MatrixXd result(10,1);
    result = n->forwardPropagation(images[24]);
    std::cout << result << std::endl;
    return 0;
}
