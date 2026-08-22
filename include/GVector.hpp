#ifndef GVECTOR_HPP
#define GVECTOR_HPP
#pragma once
#include "Maths.h"
#include "CudaKernels.cuh"
#include "CudaLaunchers.cuh"

class GVector
{
private:
    float *vector;
    size_t size;

public:
    GVector(size_t size);
    GVector();
    GVector(const GVector &other);
    GVector(GVector &&other) noexcept;
    ~GVector();
    size_t getSize();
    void setSize(size_t newSize);
    float *getVector();
    void setVector(float *newVector);
    void upload(float *vector);
    GVector &operator=(const GVector &other);
    GVector &operator=(GVector &&other) noexcept;
    GVector operator+(GVector &other);
    GVector operator-(GVector &other);
    GVector operator*(GVector &other);
    GVector operator*(float value);
    GVector operator/(double v);
    void init();
    void initZero();
    void printVec();
    void sigmoid();
};

#endif