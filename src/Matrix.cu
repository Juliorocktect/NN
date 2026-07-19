#include "Maths.h"


double crossEntropyLoss(double* y_hat,uint8_t labels);


__device__ double sigmoidDeriviative(double x)
{
    double y = sigmoid(x);
    return y * (1 - y); 
}
__device__ double sigmoid(double x){
    if (x < -100.0) x = -100.0;
    if (x >  100.0) x =  100.0;
    return 1.0 / (1.0 + std::exp(-x));
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
    }
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


    int threadsPerBlock = 512;
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
__global__ void applySigmoidDeriviative(const double* mat1,double* mat_result,int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        mat_result[idx] = sigmoidDeriviative(mat1[idx]);
    }
}
double* executeSigmoidDeriviativeKernel(const double* mat1,int rows,int cols)
{
    double* d_mat_result;
    double* d_mat1;
    size_t n = rows * cols * sizeof(double);
    int o = rows*cols;
    double* h_result = new double[o];
    cudaMalloc((void**)&d_mat_result,n);
    cudaMalloc((void**)&d_mat1,n);
    cudaMemcpy(d_mat1,mat1,n,cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (o + threadsPerBlock - 1)/threadsPerBlock;
    applySigmoidDeriviative<<<threadsPerBlock,blocksPerGrid>>>(d_mat1,d_mat_result,o);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result,d_mat1,n,cudaMemcpyDeviceToHost);
    cudaFree(&d_mat_result);
    cudaFree(&d_mat1);
    return h_result;
}
double* executeMeanMatrixKernel(double* mat,int rows,int cols)
{
    double* d_mat_result;
    double* d_mat;
    size_t size = rows * cols * sizeof(double);
    int n = rows * cols;
    double* h_res = new double[n];
    cudaMalloc((void**)& d_mat,size);
    cudaMalloc((void**)&d_mat_result,size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock -1)/threadsPerBlock;
    
    meanMatrixKernel<<<threadsPerBlock,blocksPerGrid>>>(d_mat,d_mat_result,rows,cols);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res,d_mat_result,size,cudaMemcpyDeviceToHost);

    cudaFree(&d_mat_result);
    cudaFree(&d_mat);
    return h_res;

}
__global__ void meanMatrixKernel(const double* mat,double* resultMatrix,int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        double sum = 0.0;
        for (int col = 0; col < cols; ++col)
        {
            sum += mat[row * cols + col];
        }
        resultMatrix[row] = sum / cols;
    }
}
__global__ void applyVecSoftMaxKernel(const double* mat, double* matResult, int rows, int cols)
{
    int col = blockIdx.x;
    int row = threadIdx.x;
    
    if (col < cols && row < rows)
    {
        double max;
        max = mat[ 0 * cols + col];
        for (int i = 1;i< rows;i++)//bei eins, weil 0 ist schon uf max gesetztt
        {
            if (mat[i*cols+col] > max)
            {
                max  = mat[i*cols+col];
            }            
        }
        //e^(x - max)//Warum darf auf meiner GRAKA kein double benutzen
        __shared__ double sum;
        if (row == 0) 
        {
            sum = 0.0;
            for (int i = 0; i < rows; i++) 
            {
                sum += exp(mat[i*cols + col] - max);
            }
        }
        matResult[row* cols + col] = exp(mat[row * cols + col] - max)/sum;
    }
}
double* execcuteSoftMaxKernel(double* mat, int rows,int cols)
{
    double* d_matResult;
    double* d_mat;
    double* h_matRes = new double[cols*rows];
    size_t size = rows*cols*sizeof(double);
    cudaMalloc((void**)&d_mat,size);
    cudaMalloc((void**)&d_matResult,size);

    cudaMemcpy(d_mat,mat,size,cudaMemcpyHostToDevice);
    dim3 grids(cols);
    dim3 block(rows);
    applyVecSoftMaxKernel<<< grids,block>>>(d_mat, d_matResult, rows, cols);
    cudaMemcpy(h_matRes,d_matResult,size,cudaMemcpyDeviceToHost);
    cudaFree(d_matResult);
    cudaFree(d_mat);
    return h_matRes;
}
double* executeSigmoidKernel(const double* vec,int cols,int rows)
{
    double* d_matResult;
    double* d_mat;
    size_t sizeT = cols*rows * sizeof(double);
    double* h_matResult = new double[cols*rows];
    cudaMalloc((void**)&d_mat,sizeT);
    cudaMalloc((void**)&d_matResult,sizeT);
    cudaMemcpy(d_mat,vec,sizeT,cudaMemcpyHostToDevice);
    int threads = 512;
    dim3 block(rows);
    applySigmoidToVector<<<threads,block>>>(d_mat,d_matResult,(rows*cols));

    cudaMemcpy(h_matResult,d_matResult,sizeT,cudaMemcpyDeviceToHost);
    cudaFree(d_matResult);
    cudaFree(d_mat);
    return h_matResult;
}
double* hotEncodeYMatrix(double* labels,int size)
{
    double* d_mat;
    double* d_mat_result;
    size_t sizeMat = size*sizeof(double);
    size_t sizeMatResult = size*10*sizeof(double);
    double* h_mat_result = new double[size*10];
    cudaMalloc((void**)&d_mat,sizeMat);
    cudaMalloc((void**)&d_mat_result,sizeMatResult);
    cudaMemset(d_mat_result, 0, sizeMatResult);
    cudaMemcpy(d_mat,labels,sizeMat,cudaMemcpyHostToDevice);
    int threads = 10;
    dim3 block(size);
    applyHotEncodeToMatrix<<<threads,block>>>(d_mat,d_mat_result,size*10);
    cudaMemcpy(h_mat_result,d_mat_result,sizeMatResult,cudaMemcpyDeviceToHost);
    cudaFree(d_mat);
    cudaFree(d_mat_result);
    return h_mat_result;
}
__global__ void applyHotEncodeToMatrix(double* mat,double* mat_result,int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx< size)
    {
        int label = static_cast<int>(mat[idx]);
        if(mat[idx] < 10)
        {   
            mat_result[idx * 10 + label] = 1.0;
        }
    }
}
__device__ double costF(double* vec1,double* vec2,int sizeVec1)//Labels einzeoln speichern?
{
    const double eps = 1e-15;  // verhindert log(0)

    for (int i = 0; i < sizeVec1; i++)
    {
        if (vec1[i] == 1.0)
        {
            double p = vec2[i];

            // Wahrscheinlichkeit begrenzen
            if (p < eps) p = eps;
            if (p > 1.0 - eps) p = 1.0 - eps;

            return -log(p);
        }
    }

    // Ungültiges Label (keine 1 gefunden)
    return -1.0;
}
__global__ void vectorAddMatrixKernel(double* mat,double* vec,double* mat_result,int sizeVec,int rows,int cols)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row< rows && col < cols)
    {
        mat_result[row*cols + col] = mat[row*cols +col] + vec[row];
    }

}
double* vectorAddMatrix(double* mat,double* vec,int sizeVec,int rows,int cols)
{
    double* d_mat;
    double* d_mat_result;
    double* d_vec;
    size_t sizeMat = rows*cols*sizeof(double);
    size_t sizeTVec = sizeVec*sizeof(double);
    double* h_mat_result = new double[rows*cols];
    cudaMalloc((void**)&d_mat,sizeMat);
    cudaMalloc((void**)&d_mat_result,sizeMat);
    cudaMalloc((void**)&d_vec,sizeTVec);
    cudaMemcpy(d_mat,mat,sizeMat,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec,vec,sizeTVec,cudaMemcpyHostToDevice);
    dim3 grid(cols);
    dim3 block(rows);
    vectorAddMatrixKernel<<<grid,block>>>(d_mat,d_vec,d_mat_result,sizeVec,rows,cols);
    cudaMemcpy(h_mat_result,d_mat_result,sizeMat,cudaMemcpyDeviceToHost);
    cudaFree(d_mat);
    cudaFree(d_mat_result);
    cudaFree(d_vec);
    return h_mat_result;
}
double* matrixSub(double* mat,double* mat2, int rows1,int cols1,int rows2,int cols2)
{
    if (rows1 != rows2 || cols1 != cols2)
    {
        std::cerr << "Matrixsubtraktion nicht möglich, falsche Dimensionen!" << std::endl;
        return nullptr;
    }
    double* d_mat1;
    double* d_mat2;
    double* d_mat_result;
    size_t sizeMat = rows1*cols1*sizeof(double);
    cudaMalloc((void**)&d_mat1,sizeMat);
    cudaMalloc((void**)&d_mat2,sizeMat);
    cudaMalloc((void**)&d_mat_result,sizeMat);
    cudaMemcpy(d_mat1,mat,sizeMat,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2,mat2,sizeMat,cudaMemcpyHostToDevice);
    dim3 grid(cols1);
    dim3 block(rows1);
    matrixSubKernel<<<grid,block>>>(d_mat1,d_mat2,d_mat_result,rows1,cols1);
    double* h_res = new double[rows1*cols1];
    cudaMemcpy(h_res,d_mat_result,sizeMat,cudaMemcpyDeviceToHost);
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_mat_result);
    return h_res;
}
__global__ void matrixSubKernel(double* mat1,double* mat2,double* matRes,int cols,int rows)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row< rows && col < cols)
    {
        matRes[row*cols + col] = mat1[row*cols +col] - mat2[row*cols +col];
    }
}
__global__ void vectorSubKernel(const double* vec1, const double* vec2, double* vecRes, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        vecRes[idx] = vec1[idx] - vec2[idx];
    }
}
__global__ void hadamardKernel(double* mat1,double* mat2,double* mat_result,int rows,int cols)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row< rows && col < cols)
    {
        mat_result[row*cols + col] = mat1[row*cols +col] * mat2[row*cols +col];
    }
}
double* executeHadamardKernel(double* mat,double* mat2,int row1,int col1,int row2,int col2)
{
        if (row1 != row2 || col1 != col2)
    {
        std::cerr << "Matrixsubtraktion nicht möglich, falsche Dimensionen!" << std::endl;
        return nullptr;
    }
    double* d_mat1;
    double* d_mat2;
    double* d_mat_result;
    size_t sizeMat = row1*col1*sizeof(double);
    cudaMalloc((void**)&d_mat1,sizeMat);
    cudaMalloc((void**)&d_mat2,sizeMat);
    cudaMalloc((void**)&d_mat_result,sizeMat);
    cudaMemcpy(d_mat1,mat,sizeMat,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2,mat2,sizeMat,cudaMemcpyHostToDevice);
    dim3 grid(col1);
    dim3 block(row1);
    hadamardKernel<<<grid,block>>>(d_mat1,d_mat2,d_mat_result,row1,col1);
    double* h_res = new double[row1*col1];
    cudaMemcpy(h_res,d_mat_result,sizeMat,cudaMemcpyDeviceToHost);
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_mat_result);
    return h_res;
}
double* executeMatrixDivision(double* mat1, double dividend,int rows,int cols)
{
        double* d_mat1;
    double* d_mat_result;
    size_t sizeMat = rows*cols*sizeof(double);
    cudaMalloc((void**)&d_mat1,sizeMat);
    cudaMalloc((void**)&d_mat_result,sizeMat);
    cudaMemcpy(d_mat1,mat1,sizeMat,cudaMemcpyHostToDevice);
    dim3 grid(cols);
    dim3 block(rows);
    applyMatrixDivision<<<grid,block>>>(d_mat1,d_mat_result,dividend,rows,cols);
    double* h_res = new double[rows*cols];
    cudaMemcpy(h_res,d_mat_result,sizeMat,cudaMemcpyDeviceToHost);
    cudaFree(d_mat1);
    cudaFree(d_mat_result);
    return h_res;
}
__global__ void applyMatrixDivision(double* mat,double* mat_result,double div,int rows,int cols)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row< rows && col < cols)
    {
        mat_result[row*cols + col] = mat[row*cols +col] / div;
    }
}
__global__ void singleVMatrixMultiplyKernel(double* mat,double* mat_res,double v,int rows,int cols)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row< rows && col < cols)
    {
        mat_res[row*cols + col] = mat[row*cols +col] * v;
    }
}
double* executeSingleVMatrixMultiplication(double* mat,double v, int rows,int cols)
{
    double* d_mat1;
    double* d_mat_result;
    size_t sizeMat = rows*cols*sizeof(double);
    cudaMalloc((void**)&d_mat1,sizeMat);
    cudaMalloc((void**)&d_mat_result,sizeMat);
    cudaMemcpy(d_mat1,mat,sizeMat,cudaMemcpyHostToDevice);
    dim3 grid(cols);
    dim3 block(rows);
    singleVMatrixMultiplyKernel<<<grid,block>>>(d_mat1,d_mat_result,v,rows,cols);
    double* h_res = new double[rows*cols];
    cudaMemcpy(h_res,d_mat_result,sizeMat,cudaMemcpyDeviceToHost);
    cudaFree(d_mat1);
    cudaFree(d_mat_result);
    return h_res;
}
double* executeVecSubKernel(double* vec1,double* vec2,int size)
{
    double* d_vec1;
    double* d_vec2;
    double* d_vec_res;
    double* h_res = new double[size];
    size_t sizeVec = size*sizeof(double);
    cudaMalloc((void**)&d_vec1,sizeVec);
    cudaMalloc((void**)&d_vec2,sizeVec);
    cudaMalloc((void**)&d_vec_res,sizeVec);

    cudaMemcpy(d_vec1,vec1,sizeVec,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2,vec2,sizeVec,cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksPerGrid = (size + threadsPerBlock -1)/threadsPerBlock;

    vectorSubKernel<<<threadsPerBlock,blocksPerGrid>>>(d_vec1,d_vec2,d_vec_res,size);

    cudaMemcpy(h_res,d_vec_res,sizeVec,cudaMemcpyDeviceToHost);
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_vec_res);
    return h_res;
}
__global__ void meanPerRowKernel(const double* mat, double* vec, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        double sum = 0.0;
        for (int col = 0; col < cols; ++col)
        {
            sum += mat[row * cols + col];
        }
        vec[row] = sum / cols;
    }
}
__global__ void crossEntropyLossKernel(const double* y_hat, const double* labels, double* loss, int numSamples)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples)
    {
        int label = static_cast<int>(labels[idx]);
        if (label >= 0 && label < 10)
        {
            double val = y_hat[label + idx * 10]; // Spaltenweise: label + idx*10
            if (val <= 0.0) val = 1e-8; // Schutz vor log(0)
            loss[idx] = -log(val);
        }
        else
        {
            loss[idx] = 0.0;
        }
    }
}
double executeCrossEntropyLoss(const double* y_hat, const double* labels, int numSamples)
{
    double* d_y_hat;
    double* d_labels;
    double* d_loss;
    size_t sizeYHAT = 10 * numSamples * sizeof(double);
    double* h_loss = new double[numSamples];
    
    cudaMalloc((void**)&d_y_hat,sizeYHAT);
    cudaMalloc((void**)&d_labels,numSamples*sizeof(double));
    cudaMalloc((void**)&d_loss,numSamples*sizeof(double));
    cudaMemcpy(d_y_hat,y_hat,sizeYHAT,cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels,labels,numSamples*sizeof(double),cudaMemcpyHostToDevice);
    int threadsPerBlock = 512;
    int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;
    crossEntropyLossKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y_hat, d_labels, d_loss, numSamples);
    cudaMemcpy(h_loss, d_loss, numSamples * sizeof(double), cudaMemcpyDeviceToHost);
    double sum = 0.0;
    for (int i = 0; i < numSamples; ++i)
        sum += h_loss[i];
    double meanLoss = sum / numSamples;
    delete[] h_loss;
    return meanLoss;
}   
double* executeArgmaxKernel(const double* mat,int rows, int cols)
{
    double* d_mat;
    double* h_mat_res = new double[cols];
    double* d_vec_res;
    size_t sizeVecRes = cols*sizeof(double);
    size_t sizeMat = rows * cols * sizeof(double);
    cudaMalloc((void**)&d_mat,sizeMat);
    cudaMalloc((void**)&d_vec_res,sizeVecRes);

    cudaMemcpy(d_mat,mat,sizeMat,cudaMemcpyHostToDevice);
    int threadsPerBlock = 1;
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;
    argmaxKernel<<<blocksPerGrid,threadsPerBlock>>>(d_mat,d_vec_res,rows,cols);
    cudaMemcpy(h_mat_res, d_vec_res, sizeVecRes, cudaMemcpyDeviceToHost);
    cudaFree(d_mat);
    cudaFree(d_vec_res);
    return h_mat_res;
}
__global__ void argmaxKernel(const double* mat,double* mat_result,int rows,int cols)
{
    int col = blockIdx.x;

    float maxVal = mat[col * rows];
    int index = 0;
    for (int i = 1; i < rows; i++)
    {
        
        float v = mat[col * rows + i];
        if (v > maxVal)
            maxVal = v;
            index = col*rows+i;
    }

    mat_result[col] = index;
}