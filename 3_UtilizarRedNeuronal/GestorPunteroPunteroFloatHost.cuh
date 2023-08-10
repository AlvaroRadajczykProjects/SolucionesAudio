#include <stdio.h>
#include <stdlib.h>

class GestorPunteroPunteroFloatHost {
private:
	int numero_elementos;
	int* dimensiones_elementos;
	float** h_p;
public:
	GestorPunteroPunteroFloatHost(int ne, int* de);
	~GestorPunteroPunteroFloatHost();
	int getNumeroElementos();
	int* getDimensionesElementos();
	float** getPunteroPunteroHost();
};

