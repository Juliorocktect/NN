#ifndef CUDAKERNELS_H
#define CUDAKERNELS_H
#pragma once
#include "Maths.h"
namespace CudaKernels
{
    /**
     * @brief adds the value of mat2 to mat1 on the same idx
     *
     * @param mat1
     * @param mat2
     * @param size
     * @return __global__
     */
    __global__ void matrixAddition(const float *mat1, const float *mat2, float *matResult, size_t size);

    /**
     * @brief a kernel to multiply two matrices on the GPU. The result is stored in mat3 on the VRAM
     *
     * @param mat1
     * @param mat2
     * @param mat3
     * @param rows1
     * @param cols1
     * @param cols2
     * @return __global__
     */
    __global__ void matrixMultiplication(const float *mat1, const float *mat2, float *matResult, int rows1, int cols1, int cols2);

    /**
     * @brief subtracts the value of mat2 from mat1 on the same idx
     *
     * @param mat1
     * @param mat2
     * @param size
     * @return __global__
     */
    __global__ void matrixSubstraction(const float *mat1, const float *mat2, float *matResult, size_t size);

    /**
     * @brief divides the value of mat1 by v on the same idx
     *
     * @param mat1
     * @param v
     * @param size
     * @return __global__
     */
    __global__ void matrixDivision(const float *mat1, double v, float *matResult, size_t size);

    /**
     * @brief Computes the Hadamard product of two matrices.
     *
     * @param mat1
     * @param mat2
     * @param matResult
     * @param size
     */
    __global__ void hadamardProduct(float *mat1, const float *mat2, float *matResult, size_t size);

    /**
     * @brief Adds two vectors element by element.
     *
     * @param vec1
     * @param vec2
     * @param vecResult
     * @param size
     */
    __global__ void vectorAddition(const float *vec1, const float *vec2, float *vecResult, size_t size);

    /**
     * @brief Subtracts two vectors element by element.
     *
     * @param vec1
     * @param vec2
     * @param vecResult
     * @param size
     */
    __global__ void vectorSubtraction(const float *vec1, const float *vec2, float *vecResult, size_t size);

    /**
     * @brief Computes the summed cross-entropy loss.
     *
     * @param predictions
     * @param targets
     * @param loss
     * @param size
     */
    __global__ void sumCrossEntropyLoss(const float *predictions, const float *targets, float *loss, size_t size);

    /**
     * @brief Transposes a matrix.
     *
     * @param mat
     * @param matResult
     * @param rows
     * @param cols
     */
    __global__ void transposeMatrix(const float *mat, float *matResult, int rows, int cols);

    /**
     * @brief Finds the index of the maximum value in a matrix.
     *
     * @param mat
     * @param rows
     * @param cols
     * @param result
     */
    __global__ void argmax(const float *mat, int rows, int cols, float *result);

    /**
     * @brief Computes the transposition of a matrix.
     *
     * @param input
     * @param output
     * @param rows
     * @param cols
     * @return __global__
     */
    __global__ void transposeMatrixKernel(const float *input, float *output, int rows1, int cols1, int rows2, int cols2);
    /**
     * @brief Computes the derivative of the sigmoid function.
     * Formula: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
     *
     * @param input
     * @param output
     * @param size
     * @return __global__
     */
    __global__ void sigmoidDeriviative(float *input, float *output, size_t size);
    /**
     * @brief Formula for the derivative of the sigmoid function.
     * Formula: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
     * @param x
     * @return __device__
     */
    __device__ float sigmoidDeriviativeFunction(float x);

    /**
     * @brief Matrix addition kernel for adding a vector to each row of a matrix.
     *
     * @param mat
     * @param vec
     * @param mat_result
     * @param sizeVec
     * @param rows
     * @param cols
     * @return __global__
     */
    __global__ void vectorAddMatrixKernel(float *mat, float *vec, float *mat_result, size_t sizeVec, size_t rows, size_t cols);

    /**
     * @brief Formula: sigmoid(x) = 1 / (1 + exp(-x))
     *
     * @param x
     * @return __device__
     */
    __device__ float sigmoidFunction(float x);

    /**
     * @brief Sigmoid kernel for applying the sigmoid function to each element of a matrix.
     *
     * @param input
     * @param output
     * @param size
     * @return __global__
     */
    __global__ void sigmoidKernel(float *input, float *output, size_t size);

    /**
     * @brief Softmax __device__ function for applying the softmax function to each element of a matrix.
     *
     * @param x
     * @param max
     * @param sum
     * @return __device__
     */
    __device__ float softmaxFunction(float x, float max, float sum);

    /**
     * @brief Softmax kernel for applying the softmax function to each row of a matrix.
     * Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j in the same row
     * @param input
     * @param output
     * @param rows
     * @param cols
     * @return __global__
     */
    __global__ void softmaxKernel(float *input, float *output, int rows, int cols);

    /**
     * @brief
     *
     * @param mat
     * @param matResult
     * @param size
     * @return __global__
     */
    __global__ void hotEncodeToMatrixKernel(float *mat, float *matResult, size_t size);

    /**
     * @brief
     *
     * @param y_hat
     * @param labels
     * @param loss
     * @param numSamples
     * @return __global__
     */
    __global__ void crossEntropyLossKernel(const float *y_hat, const float *labels, float *loss, int numSamples);

    /**
     * @brief Multiplies every single value of Matrix with the value
     *
     * @param mat
     * @param value
     * @param matResult
     * @param rows
     * @param cols
     * @return __global__
     */
    __global__ void matrixMultiplicationWithFloatKernel(float *mat, float value, float *matResult, size_t rows, size_t cols);
};
#endif