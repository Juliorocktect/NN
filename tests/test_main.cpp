#include <gtest/gtest.h>
#include <Maths.h>

TEST(MatrixTest,testColwiseVectorAdd) 
{
    double mat1[] = {1.2, -3.4, 5.6, 7.8, -9.1, 0.0, 2.3, -4.5, 6.7};
    double vec[] = {1.0,2.0,3.0};
    double* r = vectorAddMatrix(mat1,vec,3,3,3);
    EXPECT_DOUBLE_EQ(r[0], mat1[0]+vec[0]);
}

TEST(MatrixTest,testMatrixMultiplication) 
{
    double mat1[] = {1.2, -3.4, 5.6, 7.8, -9.1, 0.0, 2.3, -4.5, 6.7};
    double mat2[] = {3.14, -2.71, 0.577, 8.23, -4.56, 1.01, 7.77, -9.99, 5.55};
    double* r = executeMatrixMultiplicationKernel(mat1,mat2,3,3,3,3);
    double expected[] = {19.298,-43.692,28.338,-50.401,20.358,-4.690,22.246,-52.646,33.967};
    for(int i = 0;i< 9;i++)
    {
        EXPECT_NEAR(r[i], expected[i], 1e-3);
    }
}
TEST(MatrixTest, testMatrixTransposition)
{
    double mat1[] = {1.2, -3.4, 5.6,
                     7.8, -9.1, 0.0,
                     2.3, -4.5, 6.7};
    int rows = 3, cols = 3;
    double* result = executeMatrixTransposition(mat1, rows, cols);

    // Erwartete transponierte Matrix:
    double expected[] = {1.2, 7.8, 2.3,
                         -3.4, -9.1, -4.5,
                         5.6, 0.0, 6.7};

    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-6);
    }

    delete[] result;
}
TEST(MatrixTest, testMatrixMultiplicationCPU)
{
    Eigen::MatrixXd mat1(3,3);
    Eigen::MatrixXd mat2(3,3);
    mat1 << 1.2, -3.4, 5.6,
            7.8, -9.1, 0.0,
            2.3, -4.5, 6.7;
    mat2 << 3.14, -2.71, 0.577,
            8.23, -4.56, 1.01,
            7.77, -9.99, 5.55;
    Eigen::MatrixXd result;
    matrixMultiplicationCPU(mat1, mat2, result);

    Eigen::MatrixXd expected(3,3);
    expected << 19.298, -43.692, 28.338,
                -50.401, 20.358, -4.690,
                22.246, -52.646, 33.967;

    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            EXPECT_NEAR(result(i,j), expected(i,j), 1e-3);
}
TEST(MatrixTest, testMatrixSubtraction)
{
    double mat1[] = {1.2, -3.4, 5.6,
                     7.8, -9.1, 0.0,
                     2.3, -4.5, 6.7};
    double mat2[] = {3.14, -2.71, 0.577,
                     8.23, -4.56, 1.01,
                     7.77, -9.99, 5.55};
    int rows = 3, cols = 3;
    double* result = matrixSub(mat1, mat2, rows, cols, rows, cols);

    double expected[] = {
        1.2 - 3.14,   -3.4 - (-2.71),  5.6 - 0.577,
        7.8 - 8.23,   -9.1 - (-4.56),  0.0 - 1.01,
        2.3 - 7.77,   -4.5 - (-9.99),  6.7 - 5.55
    };

    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-6);
    }

    delete[] result;
}
TEST(MatrixTest, testHadamardMultiplication)
{
    double mat1[] = {1.2, -3.4, 5.6,
                     7.8, -9.1, 0.0,
                     2.3, -4.5, 6.7};
    double mat2[] = {3.14, -2.71, 0.577,
                     8.23, -4.56, 1.01,
                     7.77, -9.99, 5.55};
    int rows = 3, cols = 3;
    double* result = executeHadamardKernel(mat1, mat2, rows, cols, rows, cols);

    double expected[] = {
        1.2 * 3.14,   -3.4 * -2.71,  5.6 * 0.577,
        7.8 * 8.23,   -9.1 * -4.56,  0.0 * 1.01,
        2.3 * 7.77,   -4.5 * -9.99,  6.7 * 5.55
    };

    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-6);
    }

    delete[] result;
}
TEST(MatrixTest, testMatrixDivision)
{
    double mat1[] = {2.0, 4.0, 6.0,
                     8.0, 10.0, 12.0,
                     14.0, 16.0, 18.0};
    int rows = 3, cols = 3;
    double dividend = 2.0;
    double* result = executeMatrixDivision(mat1, dividend, rows, cols);

    double expected[] = {
        2.0 / 2.0, 4.0 / 2.0, 6.0 / 2.0,
        8.0 / 2.0, 10.0 / 2.0, 12.0 / 2.0,
        14.0 / 2.0, 16.0 / 2.0, 18.0 / 2.0
    };

    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-6);
    }

    delete[] result;
}
TEST(MatrixTest, testSingleVMatrixMultiplication)
{
    double mat[] = {1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0};
    int rows = 3, cols = 3;
    double v = 2.5;
    double* result = executeSingleVMatrixMultiplication(mat, v, rows, cols);

    double expected[] = {
        1.0 * 2.5, 2.0 * 2.5, 3.0 * 2.5,
        4.0 * 2.5, 5.0 * 2.5, 6.0 * 2.5,
        7.0 * 2.5, 8.0 * 2.5, 9.0 * 2.5
    };

    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-6);
    }

    delete[] result;
}
TEST(MatrixTest, testExecuteSingleVMatrixMultiplication)
{
    double mat[] = {1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0};
    int rows = 3, cols = 3;
    double factor = 2.5;
    double* result = executeSingleVMatrixMultiplication(mat, factor, rows, cols);

    double expected[] = {
        1.0 * 2.5, 2.0 * 2.5, 3.0 * 2.5,
        4.0 * 2.5, 5.0 * 2.5, 6.0 * 2.5,
        7.0 * 2.5, 8.0 * 2.5, 9.0 * 2.5
    };

    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-6);
    }

    delete[] result;
}
TEST(hotEncodeTest,testRepeatedLabels)
{
    double labels[] = {3, 3, 3, 3};
    int size = 4;

    double* result = hotEncodeYMatrix(labels, size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < 10; j++) {
            assert(result[i * 10 + j] == (j == 3 ? 1.0 : 0.0));
        }
    }

    delete[] result;
}
TEST(hotEncodeTest,test_all_classes)
{
    const int size = 10;
    double labels[size];

    for (int i = 0; i < size; i++)
        labels[i] = i;

    double* result = hotEncodeYMatrix(labels, size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < 10; j++) {
            assert(result[i * 10 + j] == (i == j ? 1.0 : 0.0));
        }
    }

    delete[] result;
}
TEST(hotEncodeTest,basicTest)
{
    double labels[] = {0, 2, 9};
    int size = 3;

    double* result = hotEncodeYMatrix(labels, size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < 10; j++) {
            double expected = (j == (int)labels[i]) ? 1.0 : 0.0;
            assert(result[i * 10 + j] == expected);
        }
    }

    delete[] result;
}
