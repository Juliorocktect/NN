#include "Maths.h"

__device__ double sigmoid(double x){
    if (x < -100.0) x = -100.0;
    if (x >  100.0) x =  100.0;
    return 1.0 / (1.0 + std::exp(-x));
} 
void matrixMultiplicationCPU(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2, Eigen::MatrixXd& result)
{
    if (mat1.cols() != mat2.rows()){
        std::cerr << "Matrixmultiplikation nicht möglich: falsche Dimensionen!" << std::endl;
        return;
    }
    result =  Eigen::MatrixXd(mat1.rows(), mat2.cols());
    int j = 0;//block.x
    int k = 0;//block.y
    for (int k = 0; k < mat1.rows();k++)
    {
        for (int j = 0; j < mat2.cols();j++)
        {
        double sum = 0.0;
        for (int i = 0;i < mat1.cols();i++)
        {
            double aktM1 = mat1(k,i);
            double aktM2 = mat2(i,j);
            sum += aktM1*aktM2;
        }
            result(k,j) = sum;
        }
    }
}



__global__ void vectorAdd(const double* vec1,const double* vec2,double* vec3,int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        vec3[idx] = vec2[idx] + vec1[idx];
    }
}
__global__ void matrixMultiplication(const double* mat1, const double* mat2, double* mat3, int rows1, int cols1, int cols2)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows1 && col < cols2)
    {
        double sum = 0.0;
        for (int i = 0; i < cols1; ++i)
        {
            sum += mat1[row * cols1 + i] * mat2[i * cols2 + col];
        }
        mat3[row * cols2 + col] = sum;
        printf("row: %d, col: %d, value: %f\n", row, col, sum); // Debug-Ausgabe
    }

}

 void executeFirstKernel()
{
    //1. Vectoren erstellen 
    //2. Vectoren initlisieren
    //3. auf die GPU schreiben
    Eigen::MatrixXd h_m1(10,1);
    Eigen::MatrixXd h_m2(10,1);
    double* h_result = new double[10]; // Speicher reservieren
    h_m1 << 4.0,2.0,6.5,7.8,9.1,3.4,5.6,8.2,1.0,0.5;
    h_m2 << 4.0,2.0,6.5,7.8,9.1,3.4,5.6,8.2,1.0,0.5;
    int n = 10;
    double* h_m = h_m1.data();
    double* h_m3 = h_m2.data();
    size_t size = n*sizeof(double);

    // Speicher auf der GPU anlegen
    double* d_m;
    double* d_m1;
    double* d_m2;
    cudaMalloc((void**)&d_m,size);
    cudaMalloc((void**)&d_m1,size);
    cudaMalloc((void**)&d_m2,size);
    //kopieren
    //void* h_m1_v = h_m1.data();
    cudaMemcpy(d_m,h_m,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_m1,h_m3,size,cudaMemcpyHostToDevice);
    //Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    // Kernel aufrufen
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_m,d_m1,d_m2,n);
    //zurück kopieren
    
    cudaMemcpy(h_result,d_m2,size,cudaMemcpyDeviceToHost);
    //Ausgabe
    std::cout << "Resultat: \n"<< h_m1+h_m2 << "\n";
    std::cout << "GPU berechnung:" << std::endl;
    for (int i = 0; i < n; i++){
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;
    //Speicher löschen
    cudaFree(d_m);
    cudaFree(d_m1);
    cudaFree(d_m2);
}


double* executeMatrixMultiplicationKernel(const double* mat1, const double* mat2, const int rows1, const int cols1, const int rows2, const int cols2)
{
    if (cols1 != rows2) {
        std::cerr << "Matrixmultiplikation nicht möglich: falsche Dimensionen!" << std::endl;
        return nullptr;
    }
    size_t size1 = rows1 * cols1 * sizeof(double);
    size_t size2 = rows2 * cols2 * sizeof(double);
    size_t sizeResult = rows1 * cols2 * sizeof(double);
    double* h_matResult = new double[rows1 * cols2];

    double *d_mat1, *d_mat2, *d_matResult;
    cudaMalloc((void**)&d_mat1, size1);
    cudaMalloc((void**)&d_mat2, size2);
    cudaMalloc((void**)&d_matResult, sizeResult);

    cudaMemcpy(d_mat1, mat1, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, size2, cudaMemcpyHostToDevice);

    //dimensionen festlegen
    dim3 threadsPerBlock(16, 16); //256
    dim3 blocksPerGrid((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_mat1, d_mat2, d_matResult, rows1, cols1, cols2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_matResult, d_matResult, sizeResult, cudaMemcpyDeviceToHost);

    //Man muss die Anordnung der matrix wiederherstellen
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_matResult);
    return h_matResult;
}

__global__ void MatrixAdd(const double* mat1,const double* mat2,double* matResult,int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matResult[idx] = mat1[idx] + mat2[idx];
    }
}



double* executeMatrixAdditionKernel(const double* mat1,const double* mat2,const int rows1,const int cols1,int rows2,int cols2)
{
    if (rows1 != rows2 || cols1 != cols2)
    {
        std::cerr << "Matrixaddition nicht möglich, falsche Dimensionen!" << std::endl;
        return nullptr;
    }
    double* d_mat1;
    double* d_mat2;
    double* d_matResult;
    size_t sizeMat = rows1 * cols1 * sizeof(double);
    double* h_result = new double[sizeMat];
    cudaMalloc((void**)&d_mat1,sizeMat);
    cudaMalloc((void**)&d_mat2,sizeMat);
    cudaMalloc((void**)&d_matResult,sizeMat);

    cudaMemcpy(d_mat1,mat1,sizeMat,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2,mat2,sizeMat,cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocksPerGrid = (sizeMat + threadsPerBlock -1)/threadsPerBlock;
    
    MatrixAdd<<<blocksPerGrid,threadsPerBlock>>>(d_mat1,d_mat2,d_matResult,sizeMat);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result,d_matResult,sizeMat,cudaMemcpyDeviceToHost);
    cudaFree(&d_mat1);
    cudaFree(&d_mat2);
    cudaFree(&d_matResult);
    return h_result;

}
__global__ void applySigmoidToVector(const double* vec1,double* h_resVec,int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        h_resVec[idx] = 0; sigmoid(vec1[idx]);
    }
}

double* executeMatrixTransposition(const double* mat1,int rows,int cols)
{
    double* d_mat1;
    double* d_matResult;
    size_t size = rows* cols * sizeof(double);
    double* h_matResult = new double[size];
    cudaMalloc((void**)&d_mat1,size);
    cudaMalloc((void**)&d_matResult,size);

    cudaMemcpy(d_mat1,mat1,size,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((size + threadsPerBlock.x -1) / threadsPerBlock.x,(size + threadsPerBlock.y -1)/threadsPerBlock.y);

    transposeMatrix<<<threadsPerBlock,blocksPerGrid>>>(d_mat1,d_matResult,rows,cols,cols,rows);

    cudaMemcpy(h_matResult,d_matResult,size,cudaMemcpyDeviceToHost);
    cudaFree(&d_mat1);
    cudaFree(&d_matResult);
    return h_matResult;

}
__global__ void transposeMatrix(const double* mat1,double* matResult,int rows1,int cols1,int rows2,int cols2)
{
    int cols = blockIdx.x * blockDim.x + threadIdx.x;
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    if (cols < cols1 && rows < rows2)
    {
        matResult[cols + cols1 * rows] = mat1[cols * rows2 + rows];
    }
}