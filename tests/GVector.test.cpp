#include <gtest/gtest.h>
#include <GVector.hpp>

#include <cuda_runtime.h>

#include <vector>

std::vector<float> copyVectorToHost(GVector &vector)
{
    std::vector<float> result(vector.getSize());
    EXPECT_EQ(cudaMemcpy(result.data(), vector.getVector(), result.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    return result;
}

void copyVectorToDevice(GVector &vector, const std::vector<float> &values)
{
    ASSERT_EQ(values.size(), vector.getSize());
    ASSERT_EQ(cudaMemcpy(vector.getVector(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
}

TEST(GVectorTest, DefaultConstructorCreatesEmptyVector)
{
    GVector vector;

    EXPECT_EQ(vector.getSize(), 0);
    EXPECT_EQ(vector.getVector(), nullptr);
}

TEST(GVectorTest, SizedConstructorCreatesZeroInitializedVector)
{
    GVector vector(4);

    EXPECT_EQ(vector.getSize(), 4);
    EXPECT_EQ(copyVectorToHost(vector), (std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}));
}

TEST(GVectorTest, SetSizeChangesSizeAndInitializesNewStorage)
{
    GVector vector(2);
    copyVectorToDevice(vector, {1.0f, 2.0f});

    vector.setSize(3);

    EXPECT_EQ(vector.getSize(), 3);
    EXPECT_EQ(copyVectorToHost(vector), (std::vector<float>{0.0f, 0.0f, 0.0f}));
}

TEST(GVectorTest, InitZeroResetsAllValues)
{
    GVector vector(3);
    copyVectorToDevice(vector, {1.0f, -2.0f, 3.0f});

    vector.initZero();

    EXPECT_EQ(copyVectorToHost(vector), (std::vector<float>{0.0f, 0.0f, 0.0f}));
}

TEST(GVectorTest, AssignmentCopiesValuesIndependently)
{
    GVector source(3);
    copyVectorToDevice(source, {1.0f, 2.0f, 3.0f});
    GVector copy;

    copy = source;

    EXPECT_EQ(copy.getSize(), 3);
    EXPECT_EQ(copyVectorToHost(copy), (std::vector<float>{1.0f, 2.0f, 3.0f}));

    copyVectorToDevice(source, {4.0f, 5.0f, 6.0f});
    EXPECT_EQ(copyVectorToHost(copy), (std::vector<float>{1.0f, 2.0f, 3.0f}));
}

TEST(GVectorTest, AdditionAddsValuesElementwise)
{
    GVector left(3);
    GVector right(3);
    copyVectorToDevice(left, {1.0f, 2.0f, 3.0f});
    copyVectorToDevice(right, {4.0f, 5.0f, 6.0f});

    GVector result = left + right;

    EXPECT_EQ(copyVectorToHost(result), (std::vector<float>{5.0f, 7.0f, 9.0f}));
}

TEST(GVectorTest, SubtractionSubtractsValuesElementwise)
{
    GVector left(3);
    GVector right(3);
    copyVectorToDevice(left, {5.0f, 7.0f, 9.0f});
    copyVectorToDevice(right, {1.0f, 2.0f, 3.0f});

    GVector result = left - right;

    EXPECT_EQ(copyVectorToHost(result), (std::vector<float>{4.0f, 5.0f, 6.0f}));
}

TEST(GVectorTest, MultiplicationCalculatesHadamardProduct)
{
    GVector left(3);
    GVector right(3);
    copyVectorToDevice(left, {2.0f, -3.0f, 4.0f});
    copyVectorToDevice(right, {5.0f, 2.0f, -1.0f});

    GVector result = left * right;

    EXPECT_EQ(copyVectorToHost(result), (std::vector<float>{10.0f, -6.0f, -4.0f}));
}

TEST(GVectorTest, DivisionDividesEveryElement)
{
    GVector vector(3);
    copyVectorToDevice(vector, {2.0f, 4.0f, 6.0f});

    GVector result = vector / 2.0;

    EXPECT_EQ(copyVectorToHost(result), (std::vector<float>{1.0f, 2.0f, 3.0f}));
}

TEST(GVectorTest, InitCreatesValuesInExpectedRange)
{
    GVector vector(64);

    vector.init();

    for (float value : copyVectorToHost(vector))
    {
        EXPECT_GE(value, -0.1f);
        EXPECT_LE(value, 0.1f);
    }
}

TEST(GVectorTest, SigmoidAppliesFunctionElementwise)
{
    GVector vector(4);
    copyVectorToDevice(vector, {0.0f, 1.0f, -1.0f, 2.0f});

    vector.sigmoid();

    const std::vector<float> actual = copyVectorToHost(vector);
    const std::vector<float> expected{
        0.5f,
        0.7310586f,
        0.2689414f,
        0.8807971f};

    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index)
    {
        EXPECT_NEAR(actual[index], expected[index], 1e-5f);
    }
}
