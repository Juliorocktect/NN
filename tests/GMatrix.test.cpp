#include <gtest/gtest.h>
#include <GMatrix.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
void copyToDevice(GVector &vector, const std::vector<float> &values)
{
    ASSERT_EQ(values.size(), vector.getSize());
    ASSERT_EQ(cudaMemcpy(vector.getVector(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
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
    GVector vector(2);
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

TEST(GMatrixTest, MatrixMultiplicationHandlesPartialBlocks)
{
    constexpr size_t rows = 17;
    constexpr size_t inner = 19;
    constexpr size_t cols = 18;
    GMatrix left(rows, inner);
    GMatrix right(inner, cols);
    std::vector<float> leftValues(rows * inner);
    std::vector<float> rightValues(inner * cols);
    for (size_t index = 0; index < leftValues.size(); ++index)
        leftValues[index] = static_cast<float>(static_cast<int>(index % 7) - 3) / 5.0f;
    for (size_t index = 0; index < rightValues.size(); ++index)
        rightValues[index] = static_cast<float>(static_cast<int>(index % 5) - 2) / 4.0f;
    copyToDevice(left, leftValues);
    copyToDevice(right, rightValues);

    GMatrix result = left * right;

    ASSERT_EQ(result.getRows(), rows);
    ASSERT_EQ(result.getCols(), cols);
    const std::vector<float> actual = copyToHost(result);
    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            float expected = 0.0f;
            for (size_t index = 0; index < inner; ++index)
                expected += leftValues[row * inner + index] * rightValues[index * cols + col];
            EXPECT_NEAR(actual[row * cols + col], expected, 1e-5f);
        }
    }
}

TEST(GMatrixTest, DivisionDividesEveryElement)
{
    GMatrix matrix(2, 2);
    copyToDevice(matrix, {2.0f, 4.0f, 6.0f, 8.0f});

    GMatrix result = matrix / 2.0;

    EXPECT_EQ(copyToHost(result), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(GMatrixTest, ElementwiseOperationsHandlePartialThreadBlocks)
{
    constexpr size_t rows = 17;
    constexpr size_t cols = 19;
    GMatrix left(rows, cols);
    GMatrix right(rows, cols);
    std::vector<float> leftValues(rows * cols);
    std::vector<float> rightValues(rows * cols);
    for (size_t index = 0; index < leftValues.size(); ++index)
    {
        leftValues[index] = static_cast<float>(static_cast<int>(index % 13) - 6);
        rightValues[index] = static_cast<float>(index % 5 + 1);
    }
    copyToDevice(left, leftValues);
    copyToDevice(right, rightValues);

    GMatrix sum = left + right;
    GMatrix difference = left - right;
    GMatrix scaled = left * 2.0f;
    GMatrix quotient = right / 2.0;
    const std::vector<float> actualSum = copyToHost(sum);
    const std::vector<float> actualDifference = copyToHost(difference);
    const std::vector<float> actualScaled = copyToHost(scaled);
    const std::vector<float> actualQuotient = copyToHost(quotient);
    for (size_t index = 0; index < leftValues.size(); ++index)
    {
        EXPECT_FLOAT_EQ(actualSum[index], leftValues[index] + rightValues[index]);
        EXPECT_FLOAT_EQ(actualDifference[index], leftValues[index] - rightValues[index]);
        EXPECT_FLOAT_EQ(actualScaled[index], leftValues[index] * 2.0f);
        EXPECT_FLOAT_EQ(actualQuotient[index], rightValues[index] / 2.0f);
    }
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

TEST(GMatrixTest, TransposesRectangularMatricesAcrossBlockBoundaries)
{
    constexpr size_t rows = 17;
    constexpr size_t cols = 19;
    GMatrix matrix(rows, cols);
    std::vector<float> values(rows * cols);
    for (size_t index = 0; index < values.size(); ++index)
        values[index] = static_cast<float>(index + 1);
    copyToDevice(matrix, values);

    GMatrix result = matrix.transpose();

    ASSERT_EQ(result.getRows(), cols);
    ASSERT_EQ(result.getCols(), rows);
    std::vector<float> expected(cols * rows);
    for (size_t row = 0; row < rows; ++row)
        for (size_t col = 0; col < cols; ++col)
            expected[col * rows + row] = values[row * cols + col];
    EXPECT_EQ(copyToHost(result), expected);
}

TEST(GMatrixTest, SoftmaxNormalizesEachColumn)
{
    GMatrix matrix(2, 2);
    copyToDevice(matrix, {1.0f, 2.0f, 3.0f, 4.0f});

    GMatrix result = matrix.softmax();

    const std::vector<float> actual = copyToHost(result);
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

TEST(GMatrixTest, SoftmaxHandlesRowsAcrossThreadBlockBoundary)
{
    constexpr size_t rows = 257;
    constexpr size_t cols = 17;
    GMatrix matrix(rows, cols);
    std::vector<float> values(rows * cols);
    for (size_t row = 0; row < rows; ++row)
        for (size_t col = 0; col < cols; ++col)
            values[row * cols + col] = static_cast<float>((row + col) % 11) / 5.0f;
    copyToDevice(matrix, values);

    GMatrix result = matrix.softmax();
    const std::vector<float> actual = copyToHost(result);
    for (size_t col = 0; col < cols; ++col)
    {
        float maxValue = values[col];
        for (size_t row = 1; row < rows; ++row)
            maxValue = std::max(maxValue, values[row * cols + col]);
        double sum = 0.0;
        for (size_t row = 0; row < rows; ++row)
            sum += std::exp(values[row * cols + col] - maxValue);
        for (size_t row = 0; row < rows; ++row)
        {
            const double expected = std::exp(values[row * cols + col] - maxValue) / sum;
            EXPECT_NEAR(actual[row * cols + col], expected, 1e-6);
        }
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

TEST(GMatrixTest, CalculatesMeanOfEachRow)
{
    GMatrix matrix(3, 4);
    copyToDevice(matrix, {
        1.0f, 2.0f, 3.0f, 4.0f,
        10.0f, 20.0f, 30.0f, 40.0f,
        -4.0f, -2.0f, 2.0f, 4.0f});

    GVector means = matrix.calcMeanRowise();
    std::vector<float> actual(means.getSize());
    ASSERT_EQ(cudaMemcpy(actual.data(), means.getVector(),
                         actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);

    ASSERT_EQ(actual.size(), 3);
    EXPECT_NEAR(actual[0], 2.5f, 1e-6f);
    EXPECT_NEAR(actual[1], 25.0f, 1e-6f);
    EXPECT_NEAR(actual[2], 0.0f, 1e-6f);
}

TEST(GMatrixTest, CalculatesMeansAcrossMoreThanOneThreadBlockOfRows)
{
    constexpr size_t rows = 257;
    constexpr size_t cols = 3;
    GMatrix matrix(rows, cols);
    std::vector<float> values(rows * cols);
    std::vector<float> expected(rows);
    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            values[row * cols + col] = static_cast<float>(row) + static_cast<float>(col) * 0.5f;
            expected[row] += values[row * cols + col] / static_cast<float>(cols);
        }
    }
    copyToDevice(matrix, values);

    GVector means = matrix.calcMeanRowise();
    std::vector<float> actual(rows);
    ASSERT_EQ(cudaMemcpy(actual.data(), means.getVector(), rows * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t row = 0; row < rows; ++row)
        EXPECT_NEAR(actual[row], expected[row], 1e-5f);
}

TEST(GMatrixTest, RowVectorAdditionHandlesMoreRowsThanThreadsPerBlock)
{
    constexpr size_t rows = 1025;
    constexpr size_t cols = 3;
    GMatrix matrix(rows, cols);
    GVector vector(rows);
    std::vector<float> matrixValues(rows * cols);
    std::vector<float> vectorValues(rows);
    std::vector<float> expected(rows * cols);
    for (size_t row = 0; row < rows; ++row)
    {
        vectorValues[row] = static_cast<float>(row) / 10.0f;
        for (size_t col = 0; col < cols; ++col)
        {
            const size_t index = row * cols + col;
            matrixValues[index] = static_cast<float>(index) / 20.0f;
            expected[index] = matrixValues[index] + vectorValues[row];
        }
    }
    copyToDevice(matrix, matrixValues);
    copyToDevice(vector, vectorValues);

    GMatrix result = matrix + vector;

    EXPECT_EQ(copyToHost(result), expected);
}

TEST(GMatrixTest, MultiplyWithFloatMultipliesEveryElement)
{
    GMatrix matrix(2, 2);
    copyToDevice(matrix, {1.0f, -2.0f, 0.5f, 4.0f});

    float multiplier = 2.0f;

    GMatrix result = matrix * multiplier;

    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 2);
    result.printMat();
    EXPECT_EQ(copyToHost(result), (std::vector<float>{2.0f, -4.0f, 1.0f, 8.0f}));
}

TEST(GMatrixTest, UploadMatrixToGPUCopiesHostValuesToDeviceBuffer)
{
    GMatrix matrix(2, 3);
    std::vector<float> hostBuffer = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    matrix.uploadMatrixToGPU(hostBuffer.data());

    EXPECT_EQ(copyToHost(matrix), hostBuffer);
}
