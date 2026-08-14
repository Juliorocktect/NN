#include "GMatrix.hpp"

GMatrix::GMatrix() : matrix(nullptr), rows(0), cols(0) {}

GMatrix::GMatrix(int pRows, int pCols) : matrix(nullptr), rows(pRows), cols(pCols)
{
    cudaMalloc(&matrix, rows * cols * sizeof(float));
    cudaMemset(matrix, 0, rows * cols * sizeof(float));
    // malloc mem on gpu
}
GMatrix::~GMatrix()
{
    cudaFree(&matrix);
}
size_t GMatrix::getRows()
{
    return rows;
}
size_t GMatrix::getCols()
{
    return cols;
}
float *GMatrix::getMatrix()
{
    return matrix;
}
void GMatrix::printMat()
{
    const size_t size = rows * cols;
    std::vector<float> hostMatrix(size);

    cudaError_t error = cudaMemcpy(
        hostMatrix.data(),
        matrix,
        size * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        std::cerr << "Fehler beim Kopieren der Matrix: "
                  << cudaGetErrorString(error) << std::endl;
        return;
    }

    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            std::cout << hostMatrix[row * cols + col] << " ";
        }
        std::cout << '\n';
    }
}
void GMatrix::setCols(size_t colSize)
{
    cols = colSize;
}
void GMatrix::setRows(size_t rowSize)
{
    rows = rowSize;
}
void GMatrix::setMatrix(float *otherMatrix)
{
    matrix = otherMatrix;
}
void GMatrix::initRandom()
{
    size_t total = rows * cols;
    std::vector<float> hostMatrix(total);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

    for (size_t i = 0; i < total; ++i)
    {
        hostMatrix[i] = dis(gen);
    }

    cudaMemcpy(matrix, hostMatrix.data(), total * sizeof(float), cudaMemcpyHostToDevice);
}

GMatrix GMatrix::operator+(GMatrix &other)
{
    if (other.rows == 1 && other.cols == rows)
    {
        GMatrix result(rows, cols);
        CudaLaunchers::vectorAddMatrix(matrix, other.getMatrix(), result.getMatrix(), other.getCols(), rows, cols);
        return result;
    }

    if (other.getRows() != rows || other.getCols() != cols)
    {
        std::cerr << "Matrixaddition nicht möglich, falsche Dimensionen!" << std::endl;
        return GMatrix(0, 0);
    }

    GMatrix result(rows, cols);
    CudaLaunchers::add(matrix, other.getMatrix(), result.getMatrix(), rows * cols);
    return result;
}
GMatrix &GMatrix::operator=(const GMatrix &other)
{
    if (this == &other)
        return *this;

    size_t total = static_cast<size_t>(other.rows) * other.cols;

    float *newMatrix = nullptr;
    if (total > 0)
    {
        cudaMalloc(&newMatrix, total * sizeof(float));
        cudaMemcpy(newMatrix, other.matrix, total * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(matrix);

    matrix = newMatrix;
    rows = other.rows;
    cols = other.cols;

    return *this;
}
GMatrix GMatrix::operator*(GMatrix &other)
{
    if (cols != other.getRows())
    {
        std::cerr << "Matrixmultiplikation nicht möglich, falsche Dimensionen!" << std::endl;
        return GMatrix(0, 0);
    }
    GMatrix result(rows, other.getCols());

    CudaLaunchers::multiply(matrix, other.getMatrix(), result.getMatrix(), rows, cols, other.getCols());
    return result;
}
GMatrix GMatrix::operator*(float value)
{
    GMatrix result(rows, cols);
    CudaLaunchers::matrixMultiplicationWithFloat(matrix, value, result.getMatrix(), rows, cols);
    return result;
}

GMatrix GMatrix::operator+(GVector &other)
{
    if (other.getSize() != rows)
    {
        std::cerr << "Matrixaddition nicht möglich, falsche Dimensionen!" << std::endl;
        return GMatrix(0, 0);
    }

    GMatrix result(rows, cols);
    CudaLaunchers::vectorAddMatrix(matrix, other.getVector(), result.getMatrix(), other.getSize(), rows, cols);
    return result;
}

GMatrix GMatrix::operator-(GMatrix &other)
{
    if (other.getRows() != rows || other.getCols() != cols)
    {
        std::cerr << "Matrixsubtraktion nicht möglich, falsche Dimensionen!" << std::endl;
        return GMatrix(0, 0);
    }

    GMatrix result(rows, cols);
    CudaLaunchers::subtract(matrix, other.getMatrix(), result.getMatrix(), rows * cols);
    return result;
}

GMatrix GMatrix::operator/(double v)
{
    GMatrix result(rows, cols);
    CudaLaunchers::divide(matrix, v, result.getMatrix(), rows * cols);
    return result;
}
GMatrix GMatrix::transpose()
{
    GMatrix result(cols, rows);
    CudaLaunchers::transpose(matrix, result.getMatrix(), rows, cols);
    return result;
}

GMatrix GMatrix::sigmoidDeriviative()
{
    GMatrix result(rows, cols);
    CudaLaunchers::sigmoidDeriviative(matrix, result.getMatrix(), rows * cols);
    return result;
}
GMatrix GMatrix::hadamardMultiplication(GMatrix &other)
{
    if (rows != other.getRows() || cols != other.getCols())
    {
        std::cerr << "Hadamard-Multiplikation nicht möglich, falsche Dimensionen!" << std::endl;
        return GMatrix(0, 0);
    }

    GMatrix result(rows, cols);
    CudaLaunchers::hadamardProduct(matrix, other.getMatrix(), result.getMatrix(), rows * cols);
    return result;
}
GMatrix GMatrix::sigmoid()
{
    GMatrix result(rows, cols);
    CudaLaunchers::sigmoid(matrix, result.getMatrix(), rows * cols);
    return result;
}
void GMatrix::softmax()
{
    GMatrix result(rows, cols);
    CudaLaunchers::softmax(matrix, result.getMatrix(), rows, cols);
    matrix = result.getMatrix();
}