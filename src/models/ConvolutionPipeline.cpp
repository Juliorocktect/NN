#include "ConvolutionPipe.hpp"
#include "GMatrix.hpp"

ConvolutionPipeline::ConvolutionPipeline()
{
}

ConvolutionPipeline::ConvolutionPipeline(const ConvolutionConfig &config)
{
  this->convolutionKernel1 = GMatrix(config.kernelRowSize, config.kernelColSize);
  this->convolutionKernel1.initRandom();
}

GMatrix ConvolutionPipeline::applyConvolution(GMatrix &inputMatrix)
{
  GMatrix outputMatrix(inputMatrix.getRows() - this->convolutionKernel1.getRows() + 1,
                       inputMatrix.getCols() - this->convolutionKernel1.getCols() + 1);
  CudaLaunchers::convolutionFirstLayer(inputMatrix.getMatrix(),
                                       this->convolutionKernel1.getMatrix(),
                                       outputMatrix.getMatrix(),
                                       inputMatrix.getRows(),
                                       inputMatrix.getCols(),
                                       this->convolutionKernel1.getRows(),
                                       this->convolutionKernel1.getCols());
}

ConvolutionPipeline::~ConvolutionPipeline() {}
