#include "GVector.hpp"
#include "CudaLaunchers.cuh"

#include <iostream>
#include <random>

GVector::GVector() : vector(nullptr), size(0) {}

GVector::GVector(size_t pSize) : vector(nullptr), size(pSize)
{
    if (size > 0)
    {
        cudaMalloc(&vector, size * sizeof(float));
        initZero();
    }
}

GVector::GVector(const GVector &other) : vector(nullptr), size(other.size)
{
    if (size > 0)
    {
        cudaMalloc(&vector, size * sizeof(float));
        cudaMemcpy(vector, other.vector, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

GVector::GVector(GVector &&other) noexcept : vector(other.vector), size(other.size)
{
    other.vector = nullptr;
    other.size = 0;
}

GVector::~GVector()
{
    cudaFree(vector);
}

size_t GVector::getSize()
{
    return size;
}

void GVector::setSize(size_t newSize)
{
    if (newSize == size)
    {
        return;
    }

    float *newVector = nullptr;
    if (newSize > 0)
    {
        cudaMalloc(&newVector, newSize * sizeof(float));
        cudaMemset(newVector, 0, newSize * sizeof(float));
    }

    cudaFree(vector);
    vector = newVector;
    size = newSize;
}

float *GVector::getVector()
{
    return vector;
}

void GVector::setVector(float *newVector)
{
    if (vector != newVector)
    {
        cudaFree(vector);
        vector = newVector;
    }
}

GVector &GVector::operator=(const GVector &other)
{
    if (this == &other)
    {
        return *this;
    }

    setSize(other.size);
    if (size > 0)
    {
        cudaMemcpy(vector, other.vector, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return *this;
}

GVector &GVector::operator=(GVector &&other) noexcept
{
    if (this == &other)
    {
        return *this;
    }

    cudaFree(vector);
    vector = other.vector;
    size = other.size;
    other.vector = nullptr;
    other.size = 0;
    return *this;
}

GVector GVector::operator+(GVector &other)
{
    if (size != other.size)
    {
        std::cerr << "Vektoraddition nicht möglich, falsche Dimensionen!" << std::endl;
        return GVector();
    }

    GVector result(size);
    CudaLaunchers::vectorAddition(vector, other.getVector(), result.getVector(), size);
    return result;
}

GVector GVector::operator-(GVector &other)
{
    if (size != other.size)
    {
        std::cerr << "Vektorsubtraktion nicht möglich, falsche Dimensionen!" << std::endl;
        return GVector();
    }

    GVector result(size);
    CudaLaunchers::vectorSubtraction(vector, other.getVector(), result.getVector(), size);
    return result;
}

GVector GVector::operator*(GVector &other)
{
    if (size != other.size)
    {
        std::cerr << "Hadamard-Multiplikation nicht möglich, falsche Dimensionen!" << std::endl;
        return GVector();
    }

    GVector result(size);
    CudaLaunchers::hadamardProduct(vector, other.getVector(), result.getVector(), size);
    return result;
}

GVector GVector::operator/(double v)
{
    GVector result(size);
    CudaLaunchers::divide(vector, v, result.getVector(), size);
    return result;
}

void GVector::init()
{
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);
    std::vector<float> hostVector(size);

    for (float &value : hostVector)
    {
        value = distribution(generator);
    }

    if (size > 0)
    {
        cudaMemcpy(vector, hostVector.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    }
}

void GVector::initZero()
{
    if (size > 0)
    {
        cudaMemset(vector, 0, size * sizeof(float));
    }
}

void GVector::printVec()
{
    std::vector<float> hostVector(size);
    if (size > 0)
    {
        cudaMemcpy(hostVector.data(), vector, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    for (float value : hostVector)
    {
        std::cout << value << " ";
    }
    std::cout << '\n';
}

void GVector::sigmoid()
{
    if (size > 0)
    {
        CudaLaunchers::sigmoid(vector, vector, size);
    }
}