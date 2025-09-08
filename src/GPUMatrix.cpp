#include "Maths.h"

GPUMatrix::GPUMatrix(int rows,int cols)
{
    this->rows = rows;
    this->cols = cols;
    matrix = new double[(rows*cols)];
}

GPUMatrix::GPUMatrix()
{
}
GPUMatrix::~GPUMatrix()
{

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
GPUMatrix& GPUMatrix::operator=(const GPUMatrix& other)//Man muss hier die Inhalte kopieren unlogisch aber ok
{
    if (this != &other)
    {
        this->matrix = other.matrix;
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