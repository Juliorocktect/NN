#include "Maths.h"

GPUMatrix::GPUMatrix(int pRows,int pCols)
{
    rows = pRows;
    cols = pCols;
    matrix = new double[(rows*cols)];
}

GPUMatrix::GPUMatrix()
{
}
GPUMatrix::~GPUMatrix()
{
    delete[] matrix;
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
    if (this != &other)
    {
        // Speicher ggf. freigeben
        delete[] matrix;
        // Speicher neu allokieren
        rows = other.rows;
        cols = other.cols;
        matrix = new double[rows * cols];
        std::copy(other.matrix, other.matrix + rows * cols, matrix);
    }
    return *this;
}
void GPUMatrix::transpose()
{
    double* res = executeMatrixTransposition(matrix,rows,cols);
    matrix = res;
    int tmp = rows;
    rows = cols;
    cols = tmp;
}
void GPUMatrix::init()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);

    for (int i = 0; i < rows * cols; ++i)
    {
        matrix[i] = dis(gen);
    }
}
void GPUMatrix::initZero()
{
    matrix = {0};
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