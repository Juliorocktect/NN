#include "Maths.h"

GPUMatrix::GPUMatrix()
{

}
GPUMatrix::~GPUMatrix()
{

}

__global__ void vectorAdd(const double* vec1,const double* vec2,double* vec3,int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        vec3[idx] = vec2[idx] + vec1[idx];
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
