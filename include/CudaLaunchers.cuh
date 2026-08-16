#ifndef CUDA_LAUNCHERS_CUH
#define CUDA_LAUNCHERS_CUH
#pragma once
#include "Maths.h"
namespace CudaLaunchers
{
    /**
     * @brief Cuda launcher for matrix addition kernel
     *
     * @param mat1
     * @param mat2
     * @param matResult
     * @param size
     * @return float*
     */
    float *add(float *mat1, const float *mat2, float *matResult, size_t size);
    /**
     * @brief cuda launcher for matrix multiplication kernel
     *
     * @param mat1
     * @param mat2
     * @param matResult
     * @param rows1
     * @param cols1
     * @param cols2
     * @return float*
     */
    float *multiply(float *mat1, const float *mat2, float *matResult, int rows1, int cols1, int cols2);
    /**
     * @brief Cuda launcher for matrix subtraction kernel
     *
     * @param mat1
     * @param mat2
     * @param matResult
     * @param size
     * @return float*
     */
    float *subtract(float *mat1, const float *mat2, float *matResult, size_t size);
    /**
     * @brief Cuda launcher for matrix division kernel
     *
     * @param mat1
     * @param v
     * @param matResult
     * @param size
     * @return float*
     */
    float *divide(const float *mat1, double v, float *matResult, size_t size);
    /**
     * @brief Cuda launcher for Hadamard product kernel
     * Hadamard means each index i1 is multplied by i2 from the second matrix, so the result is a matrix of the same size as the input matrices.
     *
     * @param mat1
     * @param mat2
     * @param matResult
     * @param size
     * @return float*
     */
    void hadamardProduct(float *mat1, const float *mat2, float *matResult, size_t size);
    /**
     * @brief Cuda launcher for vector addition kernel
     *
     * @param vec1
     * @param vec2
     * @param vecResult
     * @param size
     * @return float*
     */
    float *vectorAddition(const float *vec1, const float *vec2, float *vecResult, size_t size);
    /**
     * @brief Cuda launcher for vector subtraction kernel
     *
     * @param vec1
     * @param vec2
     * @param vecResult
     * @param size
     * @return float*
     */
    void vectorSubtraction(const float *vec1, const float *vec2, float *vecResult, size_t size);
    /**
     * @brief Cuda launcher for summing cross-entropy loss kernel
     *  Sum Cross-Entropy Loss is a common loss function used in machine learning for classification tasks. It measures the difference between the predicted probability distribution and the true distribution (targets). The formula for cross-entropy loss is:
     *  Formula: L = -Σ (y * log(p) + (1 - y) * log(1 - p))
     * @param predictions
     * @param targets
     * @param loss
     * @param size
     * @return float*
     */
    float *sumCrossEntropyLoss(const float *predictions, const float *targets, float *loss, size_t size);
    /**
     * @brief Cuda launcher for matrix transpose kernel
     *
     * @param mat
     * @param matResult
     * @param rows
     * @param cols
     * @return float*
     */
    float *transposeMatrix(const float *mat, float *matResult, int rows, int cols);
    /**
     * @brief Cuda launcher for argmax kernel
     *
     *  Argmax is a function that returns the index of the maximum value in a given matrix.
     *
     * @param mat
     * @param rows
     * @param cols
     * @param result
     * @return float*
     */
    float *argmax(const float *mat, int rows, int cols, float *result);
    /**
     * @brief Cuda launcher for matrix transpose kernel
     *
     * @param mat
     * @param matResult
     * @param rows
     * @param cols
     * @return float*
     */
    float *transpose(const float *mat, float *matResult, int rows, int cols);
    /**
     * @brief Cuda launcher for sigmoid derivative kernel
     * Formula: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
     *
     * @param mat
     * @param matResult
     * @param rows
     * @param cols
     * @return float*
     */
    float *sigmoidDeriviative(float *mat, float *matResult, size_t size);
    /**
     * @brief Cuda launcher for vector addition with matrix kernel
     *
     * @param mat
     * @param vec
     * @param matResult
     * @param sizeVec
     * @param cols
     * @param rows
     */
    void vectorAddMatrix(float *mat, float *vec, float *matResult, size_t sizeVec, size_t rows, size_t cols);
    /**
     * @brief applys sigmoid function to each element of the matrix
     *
     * @param mat
     * @param matResult
     * @param size
     */
    void sigmoid(float *mat, float *matResult, size_t size);
    /**
     * @brief applys softmax function to each row of the matrix
     *
     * @param mat
     * @param rows
     * @param cols
     */
    void softmax(float *mat, float *matResult, int rows, int cols);

    /**
     * @brief returns an Matrix of the size 10xSIZE_TRAINING_DATA
     *  At the correct position of the label 1 is placed
     *  the rest is 0
     *
     * @param labels
     * @param size
     * @return float*
     */
    void hotEncodeYMatrix(float *labels, float *matResult, size_t size);
    /**
     * @brief
     *
     * @param y_hat
     * @param labels
     * @param numSamples
     * @return float
     */
    float sumCrossEntropyLoss(const float *y_hat, const float *labels, int numSamples);

    /**
     * @brief Multiplies every single value of Matrix with the value
     *
     * @param mat
     * @param value
     * @param matResult
     * @param rows
     * @param cols
     */
    void matrixMultiplicationWithFloat(float *mat, float value, float *matResult, size_t rows, size_t cols);

    /**
     * @brief multiplies vec * float
     *
     * @param vec
     * @param vecResult
     * @param value
     * @param size
     */
    void vectorValueMultiplication(float *vec, float *vecResult, float value, size_t size);

    /**
     * @brief`calcMeanFromMatrixRowise` calculates the mean value of each row in a matrix. It sums all elements in a row and divides the sum by the number of elements in that row.
     *
     * @param mat
     * @param matResult
     * @param rows
     * @param cols
     */
    void meanMatrixRowise(float *mat, float *matResult, size_t rows, size_t cols);
};

#endif