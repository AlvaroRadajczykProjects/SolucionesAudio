#include "GestorPunteroPunteroFloatHost.cuh"

GestorPunteroPunteroFloatHost::GestorPunteroPunteroFloatHost(int ne, int* de) {
	numero_elementos = ne;
	dimensiones_elementos = de;

	h_p = (float**)malloc(numero_elementos * sizeof(float*));

	for (int i = 0; i < numero_elementos; i++) {
		h_p[i] = (float*)malloc(dimensiones_elementos[i] * sizeof(float));
	}
}

GestorPunteroPunteroFloatHost::~GestorPunteroPunteroFloatHost() {
	for (int i = 0; i < numero_elementos; i++) {
		free(h_p[i]);
		h_p[i] = NULL;
	}
	free(h_p);
	h_p = NULL;
	free(dimensiones_elementos);
	dimensiones_elementos = NULL;
}

int GestorPunteroPunteroFloatHost::getNumeroElementos() {
	return numero_elementos;
}

int* GestorPunteroPunteroFloatHost::getDimensionesElementos() {
	return dimensiones_elementos;
}

float** GestorPunteroPunteroFloatHost::getPunteroPunteroHost() {
	return h_p;
}
