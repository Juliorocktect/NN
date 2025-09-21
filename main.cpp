#include <VNN.h>
#include <ImageLoading.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <Maths.h>
#include <cuda_runtime.h>


int main(int argc, char const *argv[])
{
    NNG *n  = new NNG();
    double* labels = ImagePreProcessor::readLabelsAsDouble();
    n->initilizeYMatrix(labels);
    n->setInputData(ImagePreProcessor::loadImages());
    n->forwardProp();
    n->backpropagateOutputLayer();
    n->backpropagateThirdLayer();

    /* GPUMatrix m1(3,2);
    Eigen::MatrixXd mat1(3,2);
    Eigen::MatrixXd mat2(2,3);
    mat1 << 7,5,98,32,2,8;
    mat2 << 8,90,3,2,1,9;
    double data_m1[] = {7,5,98,32,2,8};
    double data_m2[] = {8,90,3,2,1,9};
    GPUMatrix m2(2,3);
    m1.matrix = data_m1;
    m2.matrix = data_m2;
    GPUMatrix result = m1*m2;
    result.printMat();
    Eigen::MatrixXd  nr = mat1 * mat2;
    std::cout << "\n";
    result.transpose();
    result.printMat();
    std::cout << nr.transpose().eval() << std::endl;
    double matrix[9] = {2.5, 1.2, 3.7,
                        0.9, 4.1, 2.2,
                        3.3, 0.5, 1.8}; 
    double* r = execcuteSoftMaxKernel(matrix,3,3);
    GPUMatrix neu(3,3);
    std::cout << "\n";
    neu.matrix = r;
    neu.printMat(); */
    
    return 0;
}
