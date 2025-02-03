// CNN.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "CNN.h"
#include "Objects.h"
#include "Neuron.h"
#include "Model.h"
using namespace std;

int main()
{
	Model *m = new Model();
	m->connectLayers();
	return 0;
}
