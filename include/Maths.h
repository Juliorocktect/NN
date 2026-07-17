#ifndef MATHS_VNN_H
#define MATHS_VNN_H
#pragma once
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>
//TODO: Store Matrix on Graphicscard
__device__ double sigmoid(double x);
__device__ double meanPerVector(double* vec);
__device__ double costF(double* vec1,double* vec2,int sizeVec1);
__global__ void sumCrossEntropyLoss(double* mat,double* sum,int rows,int cols);
__global__ void applyVecSoftMaxKernel(const double* mat,double* matResult,int rows1,int cols1);
__global__ void meanMatrixKernel(const double* mat,double* resultMatrix,int rows, int cols);
__global__ void hadamardKernel(double* mat1,double* mat2,double* mat_result,int rows,int cols);
__global__ void applySigmoidDeriviative(const double* mat1,double* mat_result,int size);
double* executeHadamardKernel(double* mat,double* mat2,int row1,int col1,int row2,int col2);
double* execcuteSoftMaxKernel(double* mat, int rows,int cols);
double* executeMeanMatrixKernel(double* mat,int rows,int cols);//result must be a row vector
double* executeSigmoidDeriviativeKernel(const double* mat1,int rows,int cols);
__global__ void MatrixAdd(const double* mat1,const double* mat2,double* matResult,int size);
double* executeSigmoidKernel(const double* vec,int cols,int rows);
__global__ void vectorAdd(const double* vec1,const double* vec2, double* vec3,int size);
__global__ void transposeMatrix(const double* mat1,double* matResult,int rows1,int cols1,int rows2,int cols2);
void executeFirstKernel();
double* matrixSub(double* mat,double* mat2, int rows1,int cols1,int rows2,int cols2);
__global__ void matrixSubKernel(double* mat1,double* mat2,double* matRes,int cols,int rows);
__global__ void vectorAddMatrixKernel(double* mat,double* vec,double* mat_result,int sizeVec,int rows,int cols);
__global__ void applySigmoidToVector(const double* vec1,double* h_resVec,int size);
__global__ void matrixMultiplication(const double* mat1, const double* mat2, double* mat3, int rows1, int cols1, int cols2);
double* executeMatrixMultiplicationKernel(const double* mat1,const double* mat2,const int rows1,const int cols1,const int rows2,const int cols2);
double* executeMatrixAdditionKernel(const double* mat1,const double* mat2,const int rows1,const int cols1,int rows2,int cols2);
double* executeMatrixTransposition(const double* mat1,int rows,int cols);
void matrixMultiplicationCPU(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2, Eigen::MatrixXd& result);
__global__ void calculateSumCost(double* y,double* y_hat);
double* hotEncodeYMatrix(double* labels,int size);
__global__ void vectorSubKernel(const double* vec1, const double* vec2, double* vecRes, int size);
double* executeVecSubKernel(double* vec1,double* vec2,int size);
double* vectorAddMatrix(double* mat,double* vec,int sizeVec,int rows,int cols);
__global__ void vectorAddMatrixKernel(double* mat,double* vec,double* mat_result,int sizeVec,int rows,int cols);
__global__ void applyHotEncodeToMatrix(double* mat,double* mat_result,int size);
__global__ void applyMatrixDivision(double* mat,double* mat_result,double div,int rows,int cols);
double* executeMatrixDivision(double* mat1, double dividend,int rows,int cols);
double* calcMeanFromMatrix(double* mat,int rows,int cols);
__global__ void singleVMatrixMultiplyKernel(double* mat,double* mat_res,double v,int rows,int cols);
double* executeSingleVMatrixMultiplication(double* mat,double v, int rows,int cols);
__global__ void meanPerRowKernel(const double* mat, double* vec, int rows, int cols);
__global__ void crossEntropyLossKernel(const double* y_hat, const double* labels, double* loss, int numSamples);
double executeCrossEntropyLoss(const double* y_hat, const double* labels, int numSamples);
double* executeArgmaxKernel(const double* mat,int rows, int cols);
__global__ void argmaxKernel(const double* mat,double* mat_result,int rows,int cols);

class GPUMatrix
{
    private:
        GPUMatrix& cudaMultiplication();
    public:
        double* matrix;//muss als row-major gespeichert werden
        int rows;
        int cols;
        GPUMatrix();
        GPUMatrix(int pRows,int pCols);
        ~GPUMatrix();
        GPUMatrix& operator=(const GPUMatrix& other);// = Operator
        GPUMatrix operator+(GPUMatrix& other);
        GPUMatrix operator-(const GPUMatrix& other);
        GPUMatrix operator*(GPUMatrix& other);
        GPUMatrix operator/(double v);
        int getRows();
        int getCols();
        void printMat();
        GPUMatrix transpose();
        GPUMatrix sigmoidDeriviative();
        void init();
        void initZero();
        GPUMatrix sigmoid();
        GPUMatrix hadamardMultiplication(GPUMatrix& other);
        void softmax();
        void addVectorColwise(GPUMatrix &other);
        GPUMatrix calcMeanFromMatrixRowise();
        GPUMatrix multiplicationSingleV(double v);
        GPUMatrix vectorSub(GPUMatrix other);
};
#endif