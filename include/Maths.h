#ifndef MATHS_VNN_H
#define MATHS_VNN_H
#pragma once
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <iostream>
//TODO: Mittelwert einer Matrix pro spalte berechnen
//TODO: Apply Softmax function
__device__ double sigmoid(double x);
__device__ double meanPerVector(double* vec);
__global__ void applyVecSoftMaxKernel(const double* mat,double* matResult,int rows1,int cols1);
__global__ void meanMatrixKernel(const double* mat,double* resultMatrix,int rows, int cols);
__global__ void applySigmoidDeriviative(const double* mat1,double* mat_result,int size);
double* execcuteSoftMaxKernel(double* mat, int rows,int cols);
double* executeMeanMatrixKernel(double* mat,int rows,int cols);//result must be a row vector
double* executeSigmoidDeriviativeKernel(const double* mat1,int rows,int cols);
__global__ void MatrixAdd(const double* mat1,const double* mat2,double* matResult,int size);
void executeSigmoidKernel(const double* vec,double* vecResult,int size);
__global__ void vectorAdd(const double* vec1,const double* vec2, double* vec3,int size);
__global__ void transposeMatrix(const double* mat1,double* matResult,int rows1,int cols1,int rows2,int cols2); 
void executeFirstKernel();
__global__ void applySigmoidToVector(const double* vec1,double* h_resVec,int size);
__global__ void matrixMultiplication(const double* mat1, const double* mat2, double* mat3, int rows1, int cols1, int cols2);
double* executeMatrixMultiplicationKernel(const double* mat1,const double* mat2,const int rows1,const int cols1,const int rows2,const int cols2);
double* executeMatrixAdditionKernel(const double* mat1,const double* mat2,const int rows1,const int cols1,int rows2,int cols2);
double* executeMatrixTransposition(const double* mat1,int rows,int cols);
void matrixMultiplicationCPU(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2, Eigen::MatrixXd& result);

//TODO: implement elementweeise operationen sigmoid deriviative
// alles ohne Eigen schreiben
class GPUMatrix
{
    private:
        GPUMatrix& cudaMultiplication();
    public:
        double* matrix;//muss als row-major gespeichert werden
        int rows,cols;
        GPUMatrix();
        GPUMatrix(int rows,int cols);
        ~GPUMatrix();
        GPUMatrix& operator=(const GPUMatrix& other);// = Operator
        GPUMatrix operator+(GPUMatrix& other);
        GPUMatrix operator-(const GPUMatrix& other);
        GPUMatrix operator*(GPUMatrix& other);
        int getRows();
        int getCols();
        void printMat();
        void transpose();
};
#endif