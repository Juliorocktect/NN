#ifndef MATHS_VNN_H
#define MATHS_VNN_H
#pragma once
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <iostream>


__global__ void MatrixAdd(const double* mat1,const double* mat2,double* matResult,int size);

__global__ void vectorAdd(const double* vec1,const double* vec2, double* vec3,int size);
void executeFirstKernel();

__global__ void matrixMultiplication(const double* mat1, const double* mat2, double* mat3, int rows1, int cols1, int cols2);

double* executeMatrixMultiplicationKernel(const double* mat1,const double* mat2,const int rows1,const int cols1,const int rows2,const int cols2);
double* executeMatrixAdditionKernel(const double* mat1,const double* mat2,const int rows1,const int cols1,int rows2,int cols2);

void matrixMultiplicationCPU(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2, Eigen::MatrixXd& result);


class GPUMatrix
{
    private:
        GPUMatrix& cudaMultiplication();
    public:
        GPUMatrix(int rows,int cols);
        GPUMatrix(const Eigen::MatrixXd& pMat);
        ~GPUMatrix();
        GPUMatrix& operator=(const GPUMatrix& other);// = Operator
        GPUMatrix operator+(GPUMatrix& other);
        GPUMatrix operator-(const GPUMatrix& other);
        GPUMatrix operator*(GPUMatrix& other);
        int rows();
        int cols();
        double* getRowMajor();
        double* getData();
        void setMat(const Eigen::MatrixXd& newMat);
        void printMat();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat;
};

#endif