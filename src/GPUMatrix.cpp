#include "Maths.h"

GPUMatrix::GPUMatrix(int pRows,int pCols) : matrix(nullptr), rows(pRows), cols(pCols)
{
    size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    matrix = (n > 0) ? new double[n] : nullptr;
}

GPUMatrix::GPUMatrix() : matrix(nullptr), rows(0), cols(0) {}

GPUMatrix::~GPUMatrix()
{
    delete[] matrix;
    matrix = nullptr;
}
int GPUMatrix::getCols()
{
    return cols;
}

int GPUMatrix::getRows()
{
    return rows;
}
void GPUMatrix::printMat()
{
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << matrix[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }
}
//Es muss immer alles rüberkopiert werden!!!!
GPUMatrix GPUMatrix::operator*(GPUMatrix& other)
{
    double* result_data = executeMatrixMultiplicationKernel(matrix, other.matrix, rows, cols, other.rows, other.cols);
    GPUMatrix result(rows,other.cols);
    result.matrix = result_data;
    result.rows = rows;
    result.cols = other.cols;
    return result;
}

GPUMatrix GPUMatrix::operator+(GPUMatrix &other)
{
        double* result_data = executeMatrixAdditionKernel(matrix,other.matrix,rows,cols,other.rows,other.cols);
        GPUMatrix result(rows, other.cols);
        result.matrix = result_data;
        return result;  
}
GPUMatrix& GPUMatrix::operator=(const GPUMatrix& other)
{
    if (this == &other) return *this;

    size_t newTotal = static_cast<size_t>(other.rows) * static_cast<size_t>(other.cols);

    // Reallocate only if size differs
    size_t curTotal = static_cast<size_t>(rows) * static_cast<size_t>(cols);
       if (newTotal != curTotal) {
            matrix = nullptr;
        delete[] matrix;
        if (newTotal > 0) matrix = new double[newTotal];
    }
       //copy matrix pointer
	   matrix = other.matrix;

    // Copy metadata and contents
    rows = other.rows;
    cols = other.cols;

    return *this;
}
GPUMatrix GPUMatrix::operator-(const GPUMatrix& other)
{
        double* result_data = matrixSub(matrix,other.matrix,rows,cols,other.rows,other.cols);
        GPUMatrix result(rows, other.cols);
        result.matrix = result_data;
        return result;  
}
GPUMatrix GPUMatrix::transpose()
{
    double* res = executeMatrixTransposition(matrix,rows,cols);
    GPUMatrix m(cols,rows);
    m.matrix = res;
    return m;
}
void GPUMatrix::init()
{
    if (rows <= 0 || cols <= 0) return;
    size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    {
        if (!matrix) matrix = new double[total];
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);
    
    for (size_t i = 0; i < total; ++i)
    {
        matrix[i] = dis(gen);
    }
}
GPUMatrix GPUMatrix::sigmoidDeriviative()
{
    double* res = executeSigmoidDeriviativeKernel(matrix,rows,cols);
    GPUMatrix m(rows,cols);
    m.matrix = res;
    return m;
}
void GPUMatrix::initZero()
{
    size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    delete[] matrix;
    matrix = (total > 0) ? new double[total]() : nullptr;
}
GPUMatrix GPUMatrix::sigmoid()
{
    GPUMatrix m(rows,cols);
    m.matrix = executeSigmoidKernel(matrix,rows,cols);
    return m;
}
void GPUMatrix::softmax()
{
    matrix = execcuteSoftMaxKernel(matrix,rows,cols);
}
void GPUMatrix::addVectorColwise(GPUMatrix &other)
{
    if (rows == other.cols && other.rows == 1)
    {
        double* result = vectorAddMatrix(matrix,other.matrix,other.cols,rows,cols);
        matrix = result;
    }
}
GPUMatrix GPUMatrix::hadamardMultiplication(GPUMatrix& other)
{
    double* res = executeHadamardKernel(matrix,other.matrix,rows,cols,other.rows,other.cols);
    GPUMatrix m(rows,cols);
    m.matrix = res;
    return m;
}
GPUMatrix GPUMatrix::operator/(double v)
{
    double* res = executeMatrixDivision(matrix,v,rows,cols);
    GPUMatrix m(rows,cols);
    m.matrix = res;
    return m;
}
GPUMatrix GPUMatrix::calcMeanFromMatrixRowise()
{
    double* res = executeMeanMatrixKernel(matrix,rows,cols);
    GPUMatrix m(rows,cols);
    m.matrix = res;
    return m;
}
GPUMatrix GPUMatrix::multiplicationSingleV(double v)
{
    double* res = executeSingleVMatrixMultiplication(matrix,v,rows,cols);
    GPUMatrix m(rows,cols);
    m.matrix = res;
    return m;
}
GPUMatrix GPUMatrix::vectorSub(GPUMatrix other)//Versichert nicht das es kein vektor ist:)
{
    double* res = executeVecSubKernel(matrix,other.matrix,rows*cols);
    GPUMatrix m(rows,cols);
    m.matrix = res;
    return m;
}