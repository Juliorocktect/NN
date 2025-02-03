#ifndef NEURON_H
#define NEURON_H
#include <cstdlib>

class Edge;

class  Neuron {
public:
	Neuron();
	Neuron(double pWeight, Edge* pEdge);
	Neuron(double pWeigt);
	Neuron(double pWeight, size_t numEdges);
	~Neuron();

	double content;
	Edge *edge;//das müssen mehrere sein
};
#endif