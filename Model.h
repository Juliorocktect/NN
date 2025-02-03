#ifndef MODEL_H
#define MODEL_H
#define INPUT_SIZE 10
#define HIDDEN_SIZE 20
#define OUTPUT_SIZE 10
#include "Neuron.h"
#include "Objects.h"
#include <random>
#include <cstdlib>
#include <iostream>


class Model {
public:
	Model();
	~Model();
	Neuron* getInputLayer();
	void connectLayers();
private:
	double bias;
	Neuron *inputLayer;
	Neuron *hiddenLayer;
	Neuron *outPutLayer;
	double activationFunction(Neuron **neuronVector, Neuron* neuronToGoTo);
	double ReLU(double x);
};

#endif // !1
