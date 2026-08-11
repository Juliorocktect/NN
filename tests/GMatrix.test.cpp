#include <gtest/gtest.h>
#include <GMatrix.hpp>

#include <cuda_runtime.h>

#include <vector>

std::vector<float> copyToHost(GMatrix &matrix)
{
    const size_t size = matrix.getRows() * matrix.getCols();
    std::vector<float> result(size);
    EXPECT_EQ(cudaMemcpy(result.data(), matrix.getMatrix(), size * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    return result;
}

void copyToDevice(GMatrix &matrix, const std::vector<float> &values)
{
    ASSERT_EQ(values.size(), matrix.getRows() * matrix.getCols());
    ASSERT_EQ(cudaMemcpy(matrix.getMatrix(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
}

void copyVectorToDeviceForMatrixTest(GVector &vector, const std::vector<float> &values)
{
    ASSERT_EQ(values.size(), vector.getSize());
    ASSERT_EQ(cudaMemcpy(vector.getVector(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
}

TEST(GMatrixTest, DefaultConstructorCreatesEmptyMatrix)
{
    GMatrix matrix;

    EXPECT_EQ(matrix.getRows(), 0);
    EXPECT_EQ(matrix.getCols(), 0);
    EXPECT_EQ(matrix.getMatrix(), nullptr);
}

TEST(GMatrixTest, SizedConstructorAllocatesZeroInitializedMatrix)
{
    GMatrix matrix(2, 3);

    EXPECT_EQ(matrix.getRows(), 2);
    EXPECT_EQ(matrix.getCols(), 3);
    EXPECT_EQ(copyToHost(matrix), (std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
}

TEST(GMatrixTest, DimensionSettersUpdateDimensions)
{
    GMatrix matrix;

    matrix.setRows(4);
    matrix.setCols(5);

    EXPECT_EQ(matrix.getRows(), 4);
    EXPECT_EQ(matrix.getCols(), 5);
}

TEST(GMatrixTest, AssignmentCopiesValuesAndDimensions)
{
    GMatrix source(2, 2);
    copyToDevice(source, {1.0f, 2.0f, 3.0f, 4.0f});
    GMatrix copy;

    copy = source;

    EXPECT_EQ(copy.getRows(), 2);
    EXPECT_EQ(copy.getCols(), 2);
    EXPECT_EQ(copyToHost(copy), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));

    copyToDevice(source, {5.0f, 6.0f, 7.0f, 8.0f});
    EXPECT_EQ(copyToHost(copy), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(GMatrixTest, AdditionAddsMatchingMatrices)
{
    GMatrix left(2, 2);
    GMatrix right(2, 2);
    copyToDevice(left, {1.0f, 2.0f, 3.0f, 4.0f});
    copyToDevice(right, {5.0f, 6.0f, 7.0f, 8.0f});

    GMatrix result = left + right;

    EXPECT_EQ(copyToHost(result), (std::vector<float>{6.0f, 8.0f, 10.0f, 12.0f}));
}

TEST(GMatrixTest, AdditionAddsRowVectorToEachMatrixColumn)
{
    GMatrix matrix(2, 3);
    GMatrix vector(1, 2);
    copyToDevice(matrix, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    copyToDevice(vector, {10.0f, 20.0f});

    GMatrix result = matrix + vector;

    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 3);
    EXPECT_EQ(copyToHost(result), (std::vector<float>{11.0f, 12.0f, 13.0f, 24.0f, 25.0f, 26.0f}));
}

TEST(GMatrixTest, AdditionAddsGVectorValueToEachMatrixRow)
{
    GMatrix matrix(2, 3);
    GVector vector(2);
    copyToDevice(matrix, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    copyVectorToDeviceForMatrixTest(vector, {10.0f, 20.0f});

    GMatrix result = matrix + vector;

    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 3);
    EXPECT_EQ(copyToHost(result), (std::vector<float>{11.0f, 12.0f, 13.0f, 24.0f, 25.0f, 26.0f}));
}

TEST(GMatrixTest, SubtractionSubtractsMatchingMatrices)
{
    GMatrix left(2, 2);
    GMatrix right(2, 2);
    copyToDevice(left, {5.0f, 6.0f, 7.0f, 8.0f});
    copyToDevice(right, {1.0f, 2.0f, 3.0f, 4.0f});

    GMatrix result = left - right;

    EXPECT_EQ(copyToHost(result), (std::vector<float>{4.0f, 4.0f, 4.0f, 4.0f}));
}

TEST(GMatrixTest, MultiplicationMultipliesMatrices)
{
    GMatrix left(2, 3);
    GMatrix right(3, 2);
    copyToDevice(left, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    copyToDevice(right, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    GMatrix result = left * right;

    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 2);
    EXPECT_EQ(copyToHost(result), (std::vector<float>{58.0f, 64.0f, 139.0f, 154.0f}));
}

TEST(GMatrixTest, DivisionDividesEveryElement)
{
    GMatrix matrix(2, 2);
    copyToDevice(matrix, {2.0f, 4.0f, 6.0f, 8.0f});

    GMatrix result = matrix / 2.0;

    EXPECT_EQ(copyToHost(result), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(GMatrixTest, TransposeSwapsDimensionsAndValues)
{
    GMatrix matrix(2, 3);
    copyToDevice(matrix, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    GMatrix result = matrix.transpose();

    EXPECT_EQ(result.getRows(), 3);
    EXPECT_EQ(result.getCols(), 2);
    EXPECT_EQ(copyToHost(result), (std::vector<float>{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}));
}

TEST(GMatrixTest, SoftmaxNormalizesEachColumn)
{
    GMatrix matrix(2, 2);
    copyToDevice(matrix, {1.0f, 2.0f, 3.0f, 4.0f});

    matrix.softmax();

    const std::vector<float> actual = copyToHost(matrix);
    const std::vector<float> expected{
        0.11920292f,
        0.11920292f,
        0.88079708f,
        0.88079708f};

    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index)
    {
        EXPECT_NEAR(actual[index], expected[index], 1e-5f);
    }
}

TEST(GMatrixTest, SigmoidDerivativeCalculatesElementwiseDerivative)
{
    GMatrix matrix(2, 2);
    copyToDevice(matrix, {0.0f, 1.0f, -1.0f, 2.0f});

    GMatrix result = matrix.sigmoidDeriviative();

    const std::vector<float> expected{
        0.25f,
        0.19661194f,
        0.19661194f,
        0.10499358f};
    const std::vector<float> actual = copyToHost(result);

    ASSERT_EQ(result.getRows(), 2);
    ASSERT_EQ(result.getCols(), 2);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index)
    {
        EXPECT_NEAR(actual[index], expected[index], 1e-5f);
    }
}

TEST(GMatrixTest, MultiplyWithFloatMultipliesEveryElement)
{
    GMatrix matrix(2, 2);
    copyToDevice(matrix, {1.0f, -2.0f, 0.5f, 4.0f});

    float multiplier = 2.0f;

    GMatrix result = matrix * multiplier;

    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 2);
    EXPECT_EQ(copyToHost(result), (std::vector<float>{2.0f, -4.0f, 1.0f, 8.0f}));
}
