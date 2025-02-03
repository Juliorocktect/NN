#include "Objects.h"
#include "Neuron.h"


Neuron::Neuron(){
	content = 0.0;
	edge = nullptr;
}
Neuron::Neuron(double pWeight,Edge* pEdge) {
	content = pWeight;
	edge = pEdge;
}
Neuron::Neuron(double pWeight)
{
	content = pWeight;
	edge = nullptr;
}
Neuron::Neuron(double pWeight, size_t numEdges)
{
	content = pWeight;
	edge = new Edge[numEdges];
}
Neuron::~Neuron() 
{
	delete[] edge;
}

Edge::Edge(){
	weight = 0.0;
	neuron = new Neuron();
}

Edge::Edge(double pWeight,Neuron *pNeuron){
	weight = pWeight;
	neuron = pNeuron;
}
Edge::~Edge()
{
	delete neuron;
}