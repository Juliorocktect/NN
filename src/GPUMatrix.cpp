#include "Maths.h"

GPUMatrix::GPUMatrix(int rows,int cols)
{
    mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();
}
GPUMatrix::GPUMatrix(const Eigen::MatrixXd& pMat)
{
    mat = pMat;
    mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(mat);
}
GPUMatrix::~GPUMatrix()
{

}


double* GPUMatrix::getData()
{
    return mat.data();
}
int GPUMatrix::cols()
{
    return mat.cols();
}

int GPUMatrix::rows()
{
    return mat.rows();
}
void GPUMatrix::setMat(const Eigen::MatrixXd& newMat)
{
    mat = newMat;
    mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(mat);
}

void GPUMatrix::printMat()
{
    std::cout << mat <<std::endl;
}

GPUMatrix GPUMatrix::operator*(GPUMatrix& other)
{
    double* result_data = executeMatrixMultiplicationKernel(mat.data(), other.getData(), mat.rows(), mat.cols(), other.rows(), other.cols());
    GPUMatrix result(mat.rows(), other.cols());
    // Ergebnisdaten in result.mat kopieren (z.B. mit Eigen::Map)
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mapped(result_data, mat.rows(), other.cols());
    result.setMat(mapped);
    delete[] result_data; // Speicher freigeben, falls nötig
    return result;
}

GPUMatrix GPUMatrix::operator+(GPUMatrix &other)
{
//TODO: implement
    double* result_data = executeMatrixAdditionKernel(mat.data(),other.mat.data(),mat.rows(),mat.cols(),other.rows(),other.cols());
    GPUMatrix result(mat.rows(), other.cols());
    // Ergebnisdaten in result.mat kopieren (z.B. mit Eigen::Map)
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mapped(result_data, mat.rows(), other.cols());
    result.setMat(mapped);
    delete[] result_data; // Speicher freigeben, falls nötig
    return result;
}
GPUMatrix& GPUMatrix::operator=(const GPUMatrix& other)//Man muss hier die Inhalte kopieren unlogisch aber ok
{
    if (this != &other)
    {
        this->mat = other.mat;
    }
    return *this;
}