#ifndef GMATRIX_HPP
#define GMATRIX_HPP
#pragma once
#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"
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
    GMatrix(int rows, int cols);
    GMatrix();
    ~GMatrix();
    size_t getRows();
    size_t getCols();
    void setCols(size_t colSize);
    void setRows(size_t rowSize);
    void setMatrix(float *otherMatrix);
    float *getMatrix();
    GMatrix &operator=(const GMatrix &other); // = Operator
    GMatrix operator+(GMatrix &other);
    GMatrix operator-(GMatrix &other);
    GMatrix operator*(GMatrix &other);
    GMatrix operator/(double v);
    GMatrix transpose();
    GMatrix sigmoidDeriviative();
    void init();
    void initZero();
    GMatrix sigmoid();
    GMatrix hadamardMultiplication(GMatrix &other);
    void softmax();
    void printMat();
    void initRandom();
    void vectorMatrixMultiplication(GMatrix &other);
};
#endif