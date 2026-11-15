#ifndef CONVOLUTION_PIPE_HPP
#define CONVOLUTION_PIPE_HPP
#include "GMatrix.hpp"
#include "TrainingConfig.hpp"

class ConvolutionPipeline
{
private:
  GMatrix convolutionKernel1; // first kernel for convolution pipeline
public:
  GMatrix applyConvolution(GMatrix &inputMatrix);
  ConvolutionPipeline();
  ConvolutionPipeline(const ConvolutionConfig &config);
  ~ConvolutionPipeline();
};
#endif
