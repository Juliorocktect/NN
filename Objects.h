#pragma once 
#include "Neuron.h"
class Edge {
public:
	double weight;
	Neuron *neuron;
	Edge(double pWeight, Neuron *pNeuron);
	Edge();
	~Edge();
};
