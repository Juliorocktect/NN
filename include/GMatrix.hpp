#ifndef GMATRIX_HPP
#define GMATRIX_HPP
#pragma once
#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"
#include "GVector.hpp"

/**
 * @brief new Matrix class to Store Data on GPU
 *
 */
class GMatrix
{
private:
    float *matrix;
    size_t rows;
    size_t cols;

public:
    /**
     * @brief Creates a matrix with the given dimensions on the GPU.
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    GMatrix(int rows, int cols);

    /**
     * @brief Creates an empty matrix without GPU storage.
     */
    GMatrix();

    /**
     * @brief Releases the matrix storage on the GPU.
     */
    ~GMatrix();

    /**
     * @brief Returns the number of rows in the matrix.
     * @return Number of rows.
     */
    size_t getRows();

    /**
     * @brief Returns the number of columns in the matrix.
     * @return Number of columns.
     */
    size_t getCols();

    /**
     * @brief Sets the number of columns.
     * @param colSize New number of columns.
     */
    void setCols(size_t colSize);

    /**
     * @brief Sets the number of rows.
     * @param rowSize New number of rows.
     */
    void setRows(size_t rowSize);

    /**
     * @brief Replaces the GPU matrix pointer.
     * @param otherMatrix Pointer to the new GPU matrix storage.
     */
    void setMatrix(float *otherMatrix);

    /**
     * @brief Returns the pointer to the matrix storage on the GPU.
     * @return GPU matrix pointer.
     */
    float *getMatrix();

    /**
     * @brief Copies another matrix into this matrix.
     * @param other Matrix to copy.
     * @return Reference to this matrix.
     */
    GMatrix &operator=(const GMatrix &other); // = Operator

    /**
     * @brief Adds two matrices element by element.
     * @param other Matrix to add.
     * @return Resulting matrix.
     */
    GMatrix operator+(GMatrix &other);

    /**
     * @brief Adds a vector value to every element in the corresponding matrix row.
     * @param other Vector to add row-wise.
     * @return Resulting matrix.
     */
    GMatrix operator+(GVector &other);

    /**
     * @brief Subtracts two matrices element by element.
     * @param other Matrix to subtract.
     * @return Resulting matrix.
     */
    GMatrix operator-(GMatrix &other);

    /**
     * @brief Multiplies two matrices.
     * @param other Matrix to multiply with.
     * @return Resulting matrix.
     */
    GMatrix operator*(GMatrix &other);

    /**
     * @brief Multiplies matrix with single float value
     * @param other Matrix to multiply with.
     * @return Resulting matrix.
     */
    GMatrix operator*(float value);

    /**
     * @brief Divides every matrix element by a scalar.
     * @param v Scalar divisor.
     * @return Resulting matrix.
     */
    GMatrix operator/(double v);

    /**
     * @brief Transposes the matrix.
     * @return Transposed matrix.
     */
    GMatrix transpose();

    /**
     * @brief Calculates the sigmoid derivative for every matrix element.
     * @return Matrix containing the element-wise sigmoid derivatives.
     */
    GMatrix sigmoidDeriviative();

    /**
     * @brief Applies the sigmoid function to every matrix element.
     * @return Matrix containing the sigmoid values.
     */
    GMatrix sigmoid();

    /**
     * @brief Calculates the Hadamard product of two matrices.
     * @param other Matrix to multiply element by element.
     * @return Resulting matrix.
     */
    GMatrix hadamardMultiplication(GMatrix &other);

    /**
     * @brief Applies the softmax function to the matrix.
     */
    void softmax();

    /**
     * @brief Prints the matrix values.
     */
    void printMat();

    /**
     * @brief Initializes all matrix elements with random values.
     */
    void initRandom();

    /**
     * @brief Multiplies the matrix with another matrix using vector-matrix multiplication.
     * @param other Matrix to multiply with.
     */
    void vectorMatrixMultiplication(GMatrix &other);
};
#endif