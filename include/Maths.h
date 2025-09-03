#ifndef MATHS_VNN_H
#define MATHS_VNN_H
#pragma once
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <iostream>


__global__ void vectorAdd(const double* vec1,const double* vec2, double* vec3,int size);
void executeFirstKernel();



class GPUMatrix
{
    private:
        //int rows,cols;
        GPUMatrix& cudaMultiplication();
        //std::vector<std::vector<double>> matrix;
    public:
        GPUMatrix();
        ~GPUMatrix();
        GPUMatrix& operator=(const GPUMatrix& other);// = Operator
        GPUMatrix& operator+(const GPUMatrix& other);
        GPUMatrix& operator-(const GPUMatrix& other);
        GPUMatrix& operator*(const GPUMatrix& other);
        
};

#endif