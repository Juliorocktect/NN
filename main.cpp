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
    ImagePreProcessor::showImage(images[24],28,28);
    return 0;
}

/*     Eigen::Matrix<double,1,2> i;
    i << 0.85,0.25;
    Eigen::Matrix<double,2,3> w1;
    w1 << 0.1,0.2,0.3,
        0.4,0.5,0.6;
    Eigen::Matrix<double,3,2> w2;
    w2 << 0.25,0.5,0.1,0.2,0.3,0.4;
    Eigen::Matrix<double,1,3> e1 = i * w1;
    for (int i = 0; i < e1.rows();i++)
    {
        double r = sigmoid(e1(0,i));
        e1(0,i) = r; 
    }
    std::cout << e1 << std::endl;
    std::cout << e1*w2 << std::endl; */