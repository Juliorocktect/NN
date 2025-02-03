#include "Model.h"

Model::Model() {
	//connectLayers();
	inputLayer = (Neuron*)malloc(INPUT_SIZE * sizeof(Neuron));
	hiddenLayer = (Neuron*)malloc(HIDDEN_SIZE * sizeof(Neuron));
	outPutLayer = (Neuron*)malloc(OUTPUT_SIZE * sizeof(Neuron));
	bias = 0.0;
}
Model::~Model(){}
Neuron* Model::getInputLayer()
{
	return inputLayer;
}
void Model::connectLayers()
{
	std::uniform_real_distribution<double> unif(0.0, 1.0);//First Layer Connection
	std::default_random_engine re;
	int amountEdges = 0;
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		double rndmDouble = unif(re);
		inputLayer[i] = Neuron(rndmDouble, HIDDEN_SIZE);
		for (int j = 0; j < HIDDEN_SIZE; j++)
		{
			double rndmEdgeWeight = unif(re);
			double rndmNeuronWeight = unif(re);
			hiddenLayer[j] = Neuron(rndmNeuronWeight,OUTPUT_SIZE);
			inputLayer[i].edge[j].neuron = &hiddenLayer[j];
			inputLayer[i].edge->weight = rndmEdgeWeight;
			amountEdges++;
		} 
	}
	for (int i = 0; i < HIDDEN_SIZE; i++)
	{

		for (int j = 0; j < OUTPUT_SIZE; j++)
		{
			double rndmNeuronWeight = unif(re);
			double rndmEdgeWeight = unif(re);
			outPutLayer[j] = Neuron(0.0);
			hiddenLayer[i].edge[j].neuron = &outPutLayer[j];
			hiddenLayer[i].edge->weight = rndmEdgeWeight;
			amountEdges++;
		}
	}
	std::cout << amountEdges << std::endl;
}
double Model::activationFunction(Neuron** neuronVector,Neuron *neuronToGoTo)//skalar Produkt
{
	double result = 0.0;
	for (int i = 0; i < (sizeof(neuronVector[0]) / sizeof(Neuron)); i++)
	{
		if (neuronVector[0][i].edge->neuron == neuronToGoTo) {
			result += (neuronVector[0][i].content * neuronVector[0][i].edge->weight);
		}
	}
	result += bias;
	return ReLU(result);
}

double Model::ReLU(double x)
{
	if (x > 0.0)
		return x;
	return 0.0;
}
